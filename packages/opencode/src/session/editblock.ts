/**
 * Text-based edit format for local models that can't produce reliable JSON tool calls.
 *
 * Parses Aider-style SEARCH/REPLACE blocks from model text output and applies them
 * using the existing tool infrastructure (permissions, LSP, file watching, etc.).
 *
 * Format:
 *   path/to/file.ts
 *   ```typescript
 *   <<<<<<< SEARCH
 *   [exact lines to find]
 *   =======
 *   [replacement lines]
 *   >>>>>>> REPLACE
 *   ```
 *
 * Empty SEARCH section = create new file or overwrite entire file.
 *
 * Ported from: https://github.com/Aider-AI/aider/blob/main/aider/coders/editblock_coder.py
 */

import fs from "fs"
import path from "path"
import { Instance } from "@/project/instance"
import { Agent } from "@/agent/agent"
import { Session } from "@/session"
import { Identifier } from "@/id/id"
import { PermissionNext } from "@/permission/next"
import { FileTime } from "@/file/time"
import { EditTool } from "@/tool/edit"
import { WriteTool } from "@/tool/write"
import type { Tool } from "@/tool/tool"
import type { MessageV2 } from "@/session/message-v2"

// ─── Parsing ────────────────────────────────────────────────────────────────

const HEAD_RE = /^<{5,9} SEARCH>?\s*$/
const DIVIDER_RE = /^={5,9}\s*$/
const UPDATED_RE = /^>{5,9} REPLACE\s*$/
const FENCE_RE = /^```/

export interface EditBlock {
  /** Absolute path to the file */
  filename: string
  /** Lines to search for. Empty string = create/overwrite file */
  search: string
  /** Lines to replace with */
  replace: string
}

export interface ParseResult {
  blocks: EditBlock[]
  errors: string[]
}

/**
 * Try to extract a filename from a raw line.
 * Strips fences, markdown styling, and language hints.
 */
function stripFilename(line: string): string | undefined {
  let s = line.trim()
  if (!s || s === "...") return undefined

  // Fence with embedded filename: ```typescript → skip; ```filename.ts → use
  if (s.startsWith("```")) {
    const rest = s.slice(3).trim()
    if (rest && (rest.includes(".") || rest.includes("/"))) return rest
    return undefined
  }

  // Strip markdown heading markers
  s = s.replace(/^#+\s*/, "")
  // Strip inline code/bold/italic markers and quotes
  s = s.replace(/[`*_"']+/g, "")
  // Strip trailing colon
  s = s.replace(/:\s*$/, "")
  s = s.trim()

  if (!s) return undefined
  // Must look like a filename (has extension or path separator)
  if (s.includes(".") || s.includes("/") || s.includes("\\")) return s
  return undefined
}

/**
 * Search backwards through lines (up to 5 lines before beforeIdx) for a filename.
 * Mirrors Aider's find_filename() logic.
 */
function findFilename(lines: string[], beforeIdx: number): string | undefined {
  for (let j = beforeIdx - 1; j >= Math.max(0, beforeIdx - 5); j--) {
    const line = lines[j]
    const candidate = stripFilename(line)
    if (candidate) return candidate

    // Stop scanning if we've passed a blank line (not immediately adjacent)
    if (!line.trim() && j < beforeIdx - 2) break
  }
  return undefined
}

/**
 * Parse all SEARCH/REPLACE blocks from raw model text.
 *
 * Returns absolute file paths. Relative paths are resolved against Instance.directory.
 */
export function parseEditBlocks(text: string): ParseResult {
  const lines = text.split("\n")
  const blocks: EditBlock[] = []
  const errors: string[] = []
  let i = 0
  let currentFilename: string | undefined

  while (i < lines.length) {
    if (!HEAD_RE.test(lines[i].trim())) {
      i++
      continue
    }

    const headIdx = i

    // ── Find filename ──────────────────────────────────────────────────────
    const rawName = findFilename(lines, headIdx)
    const resolved = rawName
      ? path.isAbsolute(rawName)
        ? rawName
        : path.join(Instance.directory, rawName)
      : currentFilename

    if (!resolved) {
      errors.push(`Line ${headIdx + 1}: SEARCH block without a filename`)
      i++
      continue
    }
    currentFilename = resolved

    // ── Collect SEARCH text (lines until DIVIDER) ─────────────────────────
    const searchLines: string[] = []
    i++
    while (i < lines.length && !DIVIDER_RE.test(lines[i].trim())) {
      searchLines.push(lines[i])
      i++
    }

    if (i >= lines.length) {
      errors.push(`Line ${headIdx + 1}: SEARCH block missing =======`)
      break
    }
    i++ // skip DIVIDER

    // ── Collect REPLACE text (lines until UPDATED or second DIVIDER) ──────
    const replaceLines: string[] = []
    while (i < lines.length && !UPDATED_RE.test(lines[i].trim()) && !DIVIDER_RE.test(lines[i].trim())) {
      replaceLines.push(lines[i])
      i++
    }

    if (i >= lines.length) {
      errors.push(`Line ${headIdx + 1}: REPLACE block missing >>>>>>> REPLACE`)
      break
    }
    i++ // skip UPDATED marker

    // Strip trailing blank lines from SEARCH (models sometimes add them)
    while (searchLines.length > 0 && !searchLines[searchLines.length - 1].trim()) {
      searchLines.pop()
    }

    blocks.push({
      filename: resolved,
      search: searchLines.join("\n"),
      replace: replaceLines.join("\n"),
    })
  }

  // Fallback: if no SEARCH/REPLACE blocks were found, look for plain fenced code blocks
  // preceded by a filename. Models sometimes output `filename.ts\n```\ncode\n```\n` without
  // using SEARCH/REPLACE markers. Treat these as whole-file writes.
  if (blocks.length === 0) {
    let j = 0
    while (j < lines.length) {
      const fname = stripFilename(lines[j])
      if (fname && j + 1 < lines.length && FENCE_RE.test(lines[j + 1].trim())) {
        // Found filename + opening fence — collect until closing fence
        j += 2 // skip filename and opening fence
        const contentLines: string[] = []
        while (j < lines.length && !FENCE_RE.test(lines[j].trim())) {
          contentLines.push(lines[j])
          j++
        }
        if (j < lines.length) j++ // skip closing fence

        if (contentLines.length > 0) {
          const resolved = path.isAbsolute(fname)
            ? fname
            : path.join(Instance.directory, fname)
          blocks.push({
            filename: resolved,
            search: "", // empty search = whole-file write
            replace: contentLines.join("\n"),
          })
        }
        continue
      }
      j++
    }
  }

  return { blocks, errors }
}

/**
 * Strip SEARCH/REPLACE blocks from model text, leaving only the conversational parts.
 * Used to clean up the displayed text after edits are applied (the diff shows separately).
 */
export function stripEditBlocks(text: string): string {
  const lines = text.split("\n")
  const output: string[] = []
  let i = 0

  while (i < lines.length) {
    // Check if this line is a SEARCH marker
    if (HEAD_RE.test(lines[i].trim())) {
      // Walk backwards to remove the filename + opening fence before the SEARCH marker.
      // Typical pattern: [blank] filename [```lang] <<<<<<< SEARCH
      let trim = output.length
      for (let j = output.length - 1; j >= Math.max(0, output.length - 5); j--) {
        const l = output[j].trim()
        if (!l) { trim = j; continue }
        if (FENCE_RE.test(l) || stripFilename(l) !== undefined) { trim = j; continue }
        break
      }
      output.length = trim

      // Skip forward past the >>>>>>> REPLACE marker
      i++
      while (i < lines.length && !UPDATED_RE.test(lines[i].trim())) i++
      i++ // skip REPLACE marker
      // Skip optional closing fence
      if (i < lines.length && FENCE_RE.test(lines[i].trim())) i++
      continue
    }

    // Check for plain fenced code blocks preceded by a filename (fallback format).
    // Pattern: filename.ts\n```lang\n...code...\n```
    if (stripFilename(lines[i]) !== undefined && i + 1 < lines.length && FENCE_RE.test(lines[i + 1].trim())) {
      // Check that this fenced block does NOT contain SEARCH/REPLACE markers
      // (those are handled by the main stripper above).
      let hasSR = false
      for (let k = i + 2; k < lines.length && !FENCE_RE.test(lines[k].trim()); k++) {
        if (HEAD_RE.test(lines[k].trim())) { hasSR = true; break }
      }
      if (!hasSR) {
        // Remove any blank lines we already pushed before the filename
        let trim = output.length
        for (let j = output.length - 1; j >= Math.max(0, output.length - 2); j--) {
          if (!output[j].trim()) { trim = j; continue }
          break
        }
        output.length = trim
        // Skip filename + opening fence + content + closing fence
        i += 2
        while (i < lines.length && !FENCE_RE.test(lines[i].trim())) i++
        if (i < lines.length) i++ // skip closing fence
        continue
      }
    }

    output.push(lines[i])
    i++
  }

  // Clean up excessive blank lines left by stripping
  const cleaned = output.join("\n").replace(/\n{3,}/g, "\n\n").trim()
  return cleaned
}

// ─── Application ─────────────────────────────────────────────────────────────

/**
 * Apply all SEARCH/REPLACE blocks found in model text.
 * Uses EditTool / WriteTool so permission prompts, LSP, file watching all work.
 *
 * Returns a summary string for each block (success or error).
 */
export async function applyTextModeEdits(input: {
  text: string
  assistantMessage: MessageV2.Assistant
  sessionID: string
  abort: AbortSignal
  /** Session messages for tool context. EditTool/WriteTool don't use this, so [] is fine. */
  messages: MessageV2.WithParts[]
}): Promise<string[]> {
  const { blocks, errors } = parseEditBlocks(input.text)
  const results: string[] = [...errors.map((e) => `Parse error: ${e}`)]

  if (blocks.length === 0) return results

  const agent = await Agent.get(input.assistantMessage.agent)
  const editToolDef = await EditTool.init()
  const writeToolDef = await WriteTool.init()

  for (const block of blocks) {
    // No-op detection: if SEARCH and REPLACE are identical (or nearly so after
    // normalizing whitespace), the model is trying to "add" something that already
    // exists. Skip the edit and report it — don't waste a tool call or permission prompt.
    if (block.search.trim()) {
      const normSearch = block.search.replace(/\s+/g, " ").trim()
      const normReplace = block.replace.replace(/\s+/g, " ").trim()
      if (normSearch === normReplace) {
        const rel = path.relative(Instance.directory, block.filename)
        results.push(`⊘ ${rel}: No change needed — the code already exists as written`)
        continue
      }
    }

    const isNewFile = !block.search.trim()
    const toolName = isNewFile ? "write" : "edit"
    const callID = Identifier.ascending("part")
    const startTime = Date.now()

    // ── Create a tool part so the TUI shows the edit ─────────────────────
    const partInput = isNewFile
      ? { filePath: block.filename, content: block.replace }
      : { filePath: block.filename, oldString: block.search, newString: block.replace }

    const part = (await Session.updatePart({
      id: Identifier.ascending("part"),
      messageID: input.assistantMessage.id,
      sessionID: input.assistantMessage.sessionID,
      type: "tool",
      tool: toolName,
      callID,
      state: {
        status: "running" as const,
        input: partInput,
        time: { start: startTime },
      },
    })) as MessageV2.ToolPart

    // ── Build Tool.Context ────────────────────────────────────────────────
    const ctx: Tool.Context = {
      sessionID: input.sessionID,
      messageID: input.assistantMessage.id,
      agent: input.assistantMessage.agent,
      abort: input.abort,
      callID,
      messages: input.messages,
      metadata({ title, metadata: metaData }) {
        // Update the running part with title/metadata from the tool (e.g. diff preview).
        Session.updatePart({
          ...part,
          state: {
            status: "running" as const,
            input: partInput,
            time: { start: startTime },
            title: title ?? undefined,
            metadata: metaData ?? undefined,
          },
        }).catch(() => {})
      },
      async ask(req) {
        await PermissionNext.ask({
          ...req,
          sessionID: input.sessionID,
          ruleset: agent.permission,
        })
      },
    }

    try {
      // Pre-register file as read so FileTime.assert passes.
      // For edits, the SEARCH block provides the safety check (content must match).
      // For whole-file writes of existing files, we still need to register the read.
      if (!isNewFile || fs.existsSync(block.filename)) {
        FileTime.read(input.sessionID, block.filename)
      }

      const result = isNewFile
        ? await writeToolDef.execute({ filePath: block.filename, content: block.replace }, ctx)
        : await editToolDef.execute({ filePath: block.filename, oldString: block.search, newString: block.replace }, ctx)

      await Session.updatePart({
        ...part,
        state: {
          status: "completed" as const,
          input: partInput,
          output: result.output,
          title: result.title,
          metadata: result.metadata,
          time: {
            start: startTime,
            end: Date.now(),
          },
        },
      })
      results.push(`✓ ${path.relative(Instance.directory, block.filename)}: ${result.output}`)
    } catch (error: any) {
      await Session.updatePart({
        ...part,
        state: {
          status: "error" as const,
          input: partInput,
          error: error.message,
          time: {
            start: startTime,
            end: Date.now(),
          },
        },
      })
      results.push(`✗ ${path.relative(Instance.directory, block.filename)}: ${error.message}`)
    }
  }

  return results
}
