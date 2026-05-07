import type { Message } from "@langchain/langgraph-sdk";

/**
 * Extracts a string summary from a message's content, supporting multimodal (text, image, file, etc.).
 * - If text is present, returns the joined text.
 * - If not, returns a label for the first non-text modality (e.g., 'Image', 'Other').
 * - If unknown, returns 'Multimodal message'.
 */
export function getContentString(content: Message["content"]): string {
  if (typeof content === "string") return content;
  const texts = content.flatMap((item) => {
    if (typeof item === "string") return [item];
    if (typeof item === "object" && item !== null) {
      const candidate = item as { type?: string; text?: unknown; content?: unknown };
      if (typeof candidate.text === "string") return [candidate.text];
      if (typeof candidate.content === "string") return [candidate.content];
    }
    return [] as string[];
  });
  return texts.length > 0 ? texts.join("") : "";
}
