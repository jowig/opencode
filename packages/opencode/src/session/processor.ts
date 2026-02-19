import { MessageV2 } from "./message-v2"
import { Log } from "@/util/log"
import { Identifier } from "@/id/id"
import { Session } from "."
import { Agent } from "@/agent/agent"
import { Snapshot } from "@/snapshot"
import { SessionSummary } from "./summary"
import { Bus } from "@/bus"
import { SessionRetry } from "./retry"
import { SessionStatus } from "./status"
import { Plugin } from "@/plugin"
import type { Provider } from "@/provider/provider"
import { LLM } from "./llm"
import { Config } from "@/config/config"
import { SessionCompaction } from "./compaction"
import { PermissionNext } from "@/permission/next"
import { Question } from "@/question"
import { applyTextModeEdits, parseEditBlocks, stripEditBlocks } from "./editblock"
import fs from "fs"
import nodePath from "path"
import { Instance } from "@/project/instance"

export namespace SessionProcessor {
  const DOOM_LOOP_THRESHOLD = 3
  const log = Log.create({ service: "session.processor" })

  export type Info = Awaited<ReturnType<typeof create>>
  export type Result = Awaited<ReturnType<Info["process"]>>

  export function create(input: {
    assistantMessage: MessageV2.Assistant
    sessionID: string
    model: Provider.Model
    abort: AbortSignal
  }) {
    const toolcalls: Record<string, MessageV2.ToolPart> = {}
    let snapshot: string | undefined
    let blocked = false
    let attempt = 0
    let needsCompaction = false

    const result = {
      get message() {
        return input.assistantMessage
      },
      partFromToolCall(toolCallID: string) {
        return toolcalls[toolCallID]
      },
      async process(streamInput: LLM.StreamInput) {
        log.info("process")
        needsCompaction = false
        const shouldBreak = (await Config.get()).experimental?.continue_loop_on_deny !== true
        while (true) {
          try {
            let currentText: MessageV2.TextPart | undefined
            let completedText = ""
            let completedTextParts: MessageV2.TextPart[] = []
            let reasoningMap: Record<string, MessageV2.ReasoningPart> = {}

            // Text-mode streaming filter: suppress SEARCH/REPLACE blocks during
            // streaming so the user only sees chat text. When a block is detected,
            // emit a brief "editing file.ts..." indicator instead.
            const isTextMode = !streamInput.model.capabilities.toolcall
            let blockState: "chat" | "block" = "chat"
            let lineBuf = "" // accumulates partial lines
            let blockFilename = "" // last filename seen before a block
            let lastFenceOrFile = "" // track filename/fence lines to suppress
            let streamedText = "" // tracks what was actually emitted to the TUI during streaming

            const stream = await LLM.stream(streamInput)

            for await (const value of stream.fullStream) {
              input.abort.throwIfAborted()
              switch (value.type) {
                case "start":
                  SessionStatus.set(input.sessionID, { type: "busy" })
                  break

                case "reasoning-start":
                  if (value.id in reasoningMap) {
                    continue
                  }
                  const reasoningPart = {
                    id: Identifier.ascending("part"),
                    messageID: input.assistantMessage.id,
                    sessionID: input.assistantMessage.sessionID,
                    type: "reasoning" as const,
                    text: "",
                    time: {
                      start: Date.now(),
                    },
                    metadata: value.providerMetadata,
                  }
                  reasoningMap[value.id] = reasoningPart
                  await Session.updatePart(reasoningPart)
                  break

                case "reasoning-delta":
                  if (value.id in reasoningMap) {
                    const part = reasoningMap[value.id]
                    part.text += value.text
                    if (value.providerMetadata) part.metadata = value.providerMetadata
                    await Session.updatePartDelta({
                      sessionID: part.sessionID,
                      messageID: part.messageID,
                      partID: part.id,
                      field: "text",
                      delta: value.text,
                    })
                  }
                  break

                case "reasoning-end":
                  if (value.id in reasoningMap) {
                    const part = reasoningMap[value.id]
                    part.text = part.text.trimEnd()

                    part.time = {
                      ...part.time,
                      end: Date.now(),
                    }
                    if (value.providerMetadata) part.metadata = value.providerMetadata
                    await Session.updatePart(part)
                    delete reasoningMap[value.id]
                  }
                  break

                case "tool-input-start":
                  const part = await Session.updatePart({
                    id: toolcalls[value.id]?.id ?? Identifier.ascending("part"),
                    messageID: input.assistantMessage.id,
                    sessionID: input.assistantMessage.sessionID,
                    type: "tool",
                    tool: value.toolName,
                    callID: value.id,
                    state: {
                      status: "pending",
                      input: {},
                      raw: "",
                    },
                  })
                  toolcalls[value.id] = part as MessageV2.ToolPart
                  break

                case "tool-input-delta":
                  break

                case "tool-input-end":
                  break

                case "tool-call": {
                  const match = toolcalls[value.toolCallId]
                  if (match) {
                    const part = await Session.updatePart({
                      ...match,
                      tool: value.toolName,
                      state: {
                        status: "running",
                        input: value.input,
                        time: {
                          start: Date.now(),
                        },
                      },
                      metadata: value.providerMetadata,
                    })
                    toolcalls[value.toolCallId] = part as MessageV2.ToolPart

                    const parts = await MessageV2.parts(input.assistantMessage.id)
                    const lastThree = parts.slice(-DOOM_LOOP_THRESHOLD)

                    if (
                      lastThree.length === DOOM_LOOP_THRESHOLD &&
                      lastThree.every(
                        (p) =>
                          p.type === "tool" &&
                          p.tool === value.toolName &&
                          p.state.status !== "pending" &&
                          JSON.stringify(p.state.input) === JSON.stringify(value.input),
                      )
                    ) {
                      const agent = await Agent.get(input.assistantMessage.agent)
                      await PermissionNext.ask({
                        permission: "doom_loop",
                        patterns: [value.toolName],
                        sessionID: input.assistantMessage.sessionID,
                        metadata: {
                          tool: value.toolName,
                          input: value.input,
                        },
                        always: [value.toolName],
                        ruleset: agent.permission,
                      })
                    }
                  }
                  break
                }
                case "tool-result": {
                  const match = toolcalls[value.toolCallId]
                  if (match && match.state.status === "running") {
                    await Session.updatePart({
                      ...match,
                      state: {
                        status: "completed",
                        input: value.input ?? match.state.input,
                        output: value.output.output,
                        metadata: value.output.metadata,
                        title: value.output.title,
                        time: {
                          start: match.state.time.start,
                          end: Date.now(),
                        },
                        attachments: value.output.attachments,
                      },
                    })

                    delete toolcalls[value.toolCallId]
                  }
                  break
                }

                case "tool-error": {
                  const match = toolcalls[value.toolCallId]
                  if (match && match.state.status === "running") {
                    await Session.updatePart({
                      ...match,
                      state: {
                        status: "error",
                        input: value.input ?? match.state.input,
                        error: (value.error as any).toString(),
                        time: {
                          start: match.state.time.start,
                          end: Date.now(),
                        },
                      },
                    })

                    if (
                      value.error instanceof PermissionNext.RejectedError ||
                      value.error instanceof Question.RejectedError
                    ) {
                      blocked = shouldBreak
                    }
                    delete toolcalls[value.toolCallId]
                  }
                  break
                }
                case "error":
                  throw value.error

                case "start-step":
                  snapshot = await Snapshot.track()
                  await Session.updatePart({
                    id: Identifier.ascending("part"),
                    messageID: input.assistantMessage.id,
                    sessionID: input.sessionID,
                    snapshot,
                    type: "step-start",
                  })
                  break

                case "finish-step":
                  const usage = Session.getUsage({
                    model: input.model,
                    usage: value.usage,
                    metadata: value.providerMetadata,
                  })
                  input.assistantMessage.finish = value.finishReason
                  input.assistantMessage.cost += usage.cost
                  input.assistantMessage.tokens = usage.tokens
                  await Session.updatePart({
                    id: Identifier.ascending("part"),
                    reason: value.finishReason,
                    snapshot: await Snapshot.track(),
                    messageID: input.assistantMessage.id,
                    sessionID: input.assistantMessage.sessionID,
                    type: "step-finish",
                    tokens: usage.tokens,
                    cost: usage.cost,
                  })
                  await Session.updateMessage(input.assistantMessage)
                  if (snapshot) {
                    const patch = await Snapshot.patch(snapshot)
                    if (patch.files.length) {
                      await Session.updatePart({
                        id: Identifier.ascending("part"),
                        messageID: input.assistantMessage.id,
                        sessionID: input.sessionID,
                        type: "patch",
                        hash: patch.hash,
                        files: patch.files,
                      })
                    }
                    snapshot = undefined
                  }
                  SessionSummary.summarize({
                    sessionID: input.sessionID,
                    messageID: input.assistantMessage.parentID,
                  })
                  if (await SessionCompaction.isOverflow({ tokens: usage.tokens, model: input.model })) {
                    needsCompaction = true
                  }
                  break

                case "text-start":
                  currentText = {
                    id: Identifier.ascending("part"),
                    messageID: input.assistantMessage.id,
                    sessionID: input.assistantMessage.sessionID,
                    type: "text",
                    text: "",
                    time: {
                      start: Date.now(),
                    },
                    metadata: value.providerMetadata,
                  }
                  await Session.updatePart(currentText)
                  break

                case "text-delta":
                  if (currentText) {
                    currentText.text += value.text
                    if (value.providerMetadata) currentText.metadata = value.providerMetadata

                    if (!isTextMode) {
                      // Normal mode: stream everything
                      await Session.updatePartDelta({
                        sessionID: currentText.sessionID,
                        messageID: currentText.messageID,
                        partID: currentText.id,
                        field: "text",
                        delta: value.text,
                      })
                    } else {
                      // Text mode: filter out SEARCH/REPLACE blocks during streaming.
                      // Buffer text and process complete lines to detect block boundaries.
                      lineBuf += value.text
                      let emitBuf = ""

                      while (lineBuf.includes("\n")) {
                        const nlIdx = lineBuf.indexOf("\n")
                        const line = lineBuf.slice(0, nlIdx)
                        lineBuf = lineBuf.slice(nlIdx + 1)
                        const trimmed = line.trim()

                        if (blockState === "chat") {
                          // Check if this line starts a SEARCH block
                          if (/^<{5,9} SEARCH>?\s*$/.test(trimmed)) {
                            blockState = "block"
                            // Emit a brief indicator so the user sees activity during block generation
                            const editLabel = blockFilename || "file"
                            emitBuf += `Editing ${editLabel}...\n`
                            continue
                          }
                          // Check if this looks like a filename before a fence
                          if (trimmed && (trimmed.includes(".") || trimmed.includes("/")) && !trimmed.startsWith("```")) {
                            // Could be a filename — hold it, emit only if next line isn't a fence/SEARCH
                            lastFenceOrFile = line + "\n"
                            blockFilename = trimmed
                            continue
                          }
                          if (/^```/.test(trimmed) && lastFenceOrFile) {
                            // Fence after filename — likely start of edit block, suppress both
                            lastFenceOrFile = ""
                            continue
                          }
                          // Regular chat line — emit it (and any held filename that wasn't a block)
                          if (lastFenceOrFile) {
                            emitBuf += lastFenceOrFile
                            lastFenceOrFile = ""
                          }
                          emitBuf += line + "\n"
                        } else {
                          // In block state: suppress everything until >>>>>>> REPLACE
                          if (/^>{5,9} REPLACE\s*$/.test(trimmed)) {
                            blockState = "chat"
                            // Skip optional closing fence on next line
                            if (lineBuf.startsWith("```")) {
                              const fenceEnd = lineBuf.indexOf("\n")
                              lineBuf = fenceEnd >= 0 ? lineBuf.slice(fenceEnd + 1) : ""
                            }
                          }
                        }
                      }

                      if (emitBuf) {
                        streamedText += emitBuf
                        await Session.updatePartDelta({
                          sessionID: currentText.sessionID,
                          messageID: currentText.messageID,
                          partID: currentText.id,
                          field: "text",
                          delta: emitBuf,
                        })
                      }
                    }
                  }
                  break

                case "text-end":
                  if (currentText) {
                    currentText.text = currentText.text.trimEnd()
                    const textOutput = await Plugin.trigger(
                      "experimental.text.complete",
                      {
                        sessionID: input.sessionID,
                        messageID: input.assistantMessage.id,
                        partID: currentText.id,
                      },
                      { text: currentText.text },
                    )
                    currentText.text = textOutput.text
                    if (value.providerMetadata) currentText.metadata = value.providerMetadata

                    // In text mode, delay setting time.end and calling updatePart
                    // until after SEARCH/REPLACE blocks are stripped. The run command
                    // only prints text parts when time.end is set, so delaying prevents
                    // raw edit blocks from appearing in output before stripping.
                    if (streamInput.model.capabilities.toolcall) {
                      currentText.time = {
                        start: currentText.time?.start ?? Date.now(),
                        end: Date.now(),
                      }
                      await Session.updatePart(currentText)
                    }
                    completedText += currentText.text + "\n"
                    completedTextParts.push({ ...currentText })
                  }
                  currentText = undefined
                  break

                case "finish":
                  break

                default:
                  log.info("unhandled", {
                    ...value,
                  })
                  continue
              }
              if (needsCompaction) break
            }

            // Text-mode iteration: handle SEARCH/REPLACE blocks and [VIEW] file requests.
            // After processing, inject a synthetic user message with results/file contents
            // and set finish="tool-calls" to keep the prompt loop going.
            if (!streamInput.model.capabilities.toolcall && completedText) {
              let continueLoop = false
              const feedbackLines: string[] = []

              // 1. Apply any SEARCH/REPLACE blocks found in the text.
              const { blocks } = parseEditBlocks(completedText)
              if (blocks.length > 0) {
                const results = await applyTextModeEdits({
                  text: completedText,
                  assistantMessage: input.assistantMessage,
                  sessionID: input.sessionID,
                  abort: input.abort,
                  messages: [],
                })
                feedbackLines.push("## Edit Results", ...results, "")
                // Only continue the loop if there were errors (so the model can retry).
                // If all edits succeeded, we're done — no need for another turn.
                const hasErrors = results.some((r) => r.startsWith("✗") || r.startsWith("Parse error"))
                if (hasErrors) {
                  continueLoop = true
                  // Include current file contents for failed files so the model can retry.
                  const failedFiles = new Set<string>()
                  for (const block of blocks) {
                    const rel = nodePath.relative(Instance.directory, block.filename)
                    if (results.some((r) => r.startsWith("✗") && r.includes(rel))) {
                      failedFiles.add(block.filename)
                    }
                  }
                  for (const abs of failedFiles) {
                    try {
                      const content = fs.readFileSync(abs, "utf-8")
                      const rel = nodePath.relative(Instance.directory, abs)
                      feedbackLines.push(`## Current contents of ${rel}`, "```", content, "```", "")
                    } catch {}
                  }
                  feedbackLines.push("Please output corrected SEARCH/REPLACE blocks to retry the failed edits.")
                }

                // Strip SEARCH/REPLACE blocks from the displayed text parts so only
                // the conversational chat remains (the diff shows as a separate tool part).
                for (const part of completedTextParts) {
                  const stripped = stripEditBlocks(part.text)
                  if (stripped !== part.text) {
                    part.text = stripped
                  }
                }

                // Fallback: if stripping left all text parts empty, the model didn't chat.
                // Preserve what was streamed to the user (e.g. "Editing calculator.ts...")
                // so completion doesn't blank out the text they already saw.
                const allEmpty = completedTextParts.every((p) => !p.text.trim())
                if (allEmpty && completedTextParts.length > 0) {
                  const trimmedStreamed = streamedText.trim()
                  if (trimmedStreamed) {
                    // Keep the text that was shown during streaming
                    completedTextParts[0].text = trimmedStreamed
                  } else if (blocks.length > 0) {
                    // Nothing was streamed either — synthesize a brief description
                    const descriptions: string[] = []
                    for (const block of blocks) {
                      const rel = nodePath.relative(Instance.directory, block.filename)
                      const normS = block.search.replace(/\s+/g, " ").trim()
                      const normR = block.replace.replace(/\s+/g, " ").trim()
                      if (block.search.trim() && normS === normR) {
                        descriptions.push(`${rel} already has the requested change`)
                      } else if (!block.search.trim()) {
                        descriptions.push(`Created ${rel}`)
                      } else {
                        descriptions.push(`Edited ${rel}`)
                      }
                    }
                    const unique = [...new Set(descriptions)]
                    completedTextParts[0].text = unique.join(". ") + "."
                  }
                }
              }

              // Finalize text parts: set time.end and publish the (possibly stripped) text.
              // This is deferred from text-end so the run command only sees the final version.
              for (const part of completedTextParts) {
                if (!part.time?.end) {
                  part.time = { start: part.time?.start ?? Date.now(), end: Date.now() }
                }
                await Session.updatePart(part)
              }

              // 2. Detect [VIEW path/to/file] requests and inject file contents.
              const viewRequests = completedText.match(/\[VIEW\s+([^\]]+)\]/g) ?? []
              for (const match of viewRequests) {
                const rawPath = match.replace(/\[VIEW\s+/, "").replace(/\]$/, "").trim()
                const abs = nodePath.isAbsolute(rawPath) ? rawPath : nodePath.join(Instance.directory, rawPath)
                try {
                  const content = fs.readFileSync(abs, "utf-8")
                  const rel = nodePath.relative(Instance.directory, abs)
                  feedbackLines.push(`## Contents of ${rel}`, "```", content, "```", "")
                  continueLoop = true
                } catch {
                  feedbackLines.push(`## ${rawPath}`, "File not found or could not be read.", "")
                  continueLoop = true
                }
              }

              // 3. Auto-inject files: if the model chatted but produced no edits and no [VIEW],
              // scan its text for file references and inject their contents. This handles the
              // case where the model says "I need to see the file" without using [VIEW] syntax.
              if (!continueLoop && blocks.length === 0 && viewRequests.length === 0) {
                const fileRefs = completedText.match(/[\w./-]+\.\w{1,10}/g) ?? []
                const seen = new Set<string>()
                for (const ref of fileRefs) {
                  if (seen.has(ref)) continue
                  seen.add(ref)
                  const abs = nodePath.isAbsolute(ref) ? ref : nodePath.join(Instance.directory, ref)
                  try {
                    const stat = fs.statSync(abs)
                    if (stat.isFile() && stat.size < 100_000) {
                      const content = fs.readFileSync(abs, "utf-8")
                      const rel = nodePath.relative(Instance.directory, abs)
                      feedbackLines.push(`## Contents of ${rel}`, "```", content, "```", "")
                      continueLoop = true
                    }
                  } catch {}
                }
                if (continueLoop) {
                  feedbackLines.push("The file contents are shown above. Please make the requested changes using SEARCH/REPLACE blocks.")
                }
              }

              // 4. If we have feedback, inject a synthetic user message and continue the loop.
              if (continueLoop && feedbackLines.length > 0) {
                input.assistantMessage.finish = "tool-calls"
                await Session.updateMessage(input.assistantMessage)

                const syntheticUser: MessageV2.User = {
                  id: Identifier.ascending("message"),
                  sessionID: input.sessionID,
                  role: "user",
                  time: { created: Date.now() },
                  agent: streamInput.user.agent,
                  model: streamInput.user.model,
                }
                await Session.updateMessage(syntheticUser)
                await Session.updatePart({
                  id: Identifier.ascending("part"),
                  messageID: syntheticUser.id,
                  sessionID: input.sessionID,
                  type: "text",
                  text: feedbackLines.join("\n"),
                  synthetic: true,
                } satisfies MessageV2.TextPart)
              }
            }
          } catch (e: any) {
            log.error("process", {
              error: e,
              stack: JSON.stringify(e.stack),
            })
            const error = MessageV2.fromError(e, { providerID: input.model.providerID })
            if (MessageV2.ContextOverflowError.isInstance(error)) {
              // TODO: Handle context overflow error
            }
            const retry = SessionRetry.retryable(error)
            if (retry !== undefined) {
              attempt++
              const delay = SessionRetry.delay(attempt, error.name === "APIError" ? error : undefined)
              SessionStatus.set(input.sessionID, {
                type: "retry",
                attempt,
                message: retry,
                next: Date.now() + delay,
              })
              await SessionRetry.sleep(delay, input.abort).catch(() => {})
              continue
            }
            input.assistantMessage.error = error
            Bus.publish(Session.Event.Error, {
              sessionID: input.assistantMessage.sessionID,
              error: input.assistantMessage.error,
            })
            SessionStatus.set(input.sessionID, { type: "idle" })
          }
          if (snapshot) {
            const patch = await Snapshot.patch(snapshot)
            if (patch.files.length) {
              await Session.updatePart({
                id: Identifier.ascending("part"),
                messageID: input.assistantMessage.id,
                sessionID: input.sessionID,
                type: "patch",
                hash: patch.hash,
                files: patch.files,
              })
            }
            snapshot = undefined
          }
          const p = await MessageV2.parts(input.assistantMessage.id)
          for (const part of p) {
            if (part.type === "tool" && part.state.status !== "completed" && part.state.status !== "error") {
              await Session.updatePart({
                ...part,
                state: {
                  ...part.state,
                  status: "error",
                  error: "Tool execution aborted",
                  time: {
                    start: Date.now(),
                    end: Date.now(),
                  },
                },
              })
            }
          }
          input.assistantMessage.time.completed = Date.now()
          await Session.updateMessage(input.assistantMessage)
          if (needsCompaction) return "compact"
          if (blocked) return "stop"
          if (input.assistantMessage.error) return "stop"
          return "continue"
        }
      },
    }
    return result
  }
}
