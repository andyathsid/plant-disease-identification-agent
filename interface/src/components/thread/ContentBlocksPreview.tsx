import React from "react";
import type { Base64ContentBlock } from "@langchain/core/messages";
import { MultimodalPreview } from "./MultimodalPreview";
import { cn } from "@/lib/utils";

interface ContentBlocksPreviewProps {
  blocks: Base64ContentBlock[];
  onRemove: (idx: number) => void;
  size?: "sm" | "md" | "lg";
  className?: string;
  uploadingFiles?: Record<string, boolean>; // Add uploadingFiles prop
}

/**
 * Renders a preview of content blocks with optional remove functionality.
 * Uses cn utility for robust class merging.
 */
export const ContentBlocksPreview: React.FC<ContentBlocksPreviewProps> = ({
  blocks,
  onRemove,
  size = "md",
  className,
  uploadingFiles = {},
}) => {
  if (!blocks.length) return null;

  return (
    <div className={cn("flex flex-wrap gap-2 p-3.5 pb-0", className)}>
      {blocks.map((block, idx) => {
        // Determine if this image is currently uploading
        let isUploading = false;
        if (block.type === "image" && block.source_type === "base64" && block.mime_type?.startsWith("image/")) {
          // Check if this specific file is currently uploading using the fileId stored in metadata
          const fileId = block.metadata?.fileId as string;
          isUploading = fileId ? !!uploadingFiles[fileId] : false;
        }

        return (
          <MultimodalPreview
            key={idx}
            block={block}
            removable
            onRemove={() => onRemove(idx)}
            size={size}
            isUploading={isUploading}
          />
        );
      })}
    </div>
  );
};
