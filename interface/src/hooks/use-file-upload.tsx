import { useState, useRef, useEffect, ChangeEvent } from "react";
import type { Base64ContentBlock } from "@langchain/core/messages";
import { validateFiles, showFileValidationErrors } from "@/lib/file-validation";
import { fileToContentBlock } from "@/lib/multimodal-utils";
import { uploadImageAction } from "@/lib/supabase-actions";

interface UseFileUploadOptions {
  initialBlocks?: Base64ContentBlock[];
}

// Define a type for tracking upload status
interface UploadStatus {
  [key: string]: boolean; // Key is the unique identifier for the file, value indicates if uploading
}

interface UseFileUploadReturn {
  contentBlocks: Base64ContentBlock[];
  setContentBlocks: React.Dispatch<React.SetStateAction<Base64ContentBlock[]>>;
  handleFileUpload: (e: ChangeEvent<HTMLInputElement>) => Promise<void>;
  dropRef: React.RefObject<HTMLDivElement | null>;
  removeBlock: (idx: number) => void;
  resetBlocks: () => void;
  dragOver: boolean;
  handlePaste: (e: React.ClipboardEvent<HTMLTextAreaElement | HTMLInputElement>) => Promise<void>;
  uploadingFiles: UploadStatus;
}

export function useFileUpload({
  initialBlocks = [],
}: UseFileUploadOptions = {}): UseFileUploadReturn {
  const [contentBlocks, setContentBlocks] =
    useState<Base64ContentBlock[]>(initialBlocks);
  const [uploadingFiles, setUploadingFiles] = useState<UploadStatus>({});
  const dropRef = useRef<HTMLDivElement | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const dragCounter = useRef(0);

  const handleFileUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    const fileArray = Array.from(files);
    const validation = validateFiles(fileArray, contentBlocks);
    showFileValidationErrors(validation, false);

    const newBlocks = await Promise.all(
      validation.uniqueFiles.map(async (file) => {
        const block = await fileToContentBlock(file);
        if (file.type.startsWith("image/")) {
          // Generate a unique ID for this file upload
          const fileId = `${file.name}-${file.size}-${file.lastModified}`;

          // Mark this file as uploading
          setUploadingFiles(prev => ({ ...prev, [fileId]: true }));

          try {
            const formData = new FormData();
            formData.append("file", file);
            const url = await uploadImageAction(formData);
            block.metadata = { ...block.metadata, url, uploading: false, fileId };
          } catch (error) {
            console.error("Failed to upload image to Supabase:", error);
            block.metadata = { ...block.metadata, uploading: false, fileId };
          } finally {
            // Mark this file as no longer uploading
            setUploadingFiles(prev => {
              const newState = { ...prev };
              delete newState[fileId];
              return newState;
            });
          }
        }
        return block;
      }),
    );

    if (newBlocks.length > 0) {
      setContentBlocks((prev) => [...prev, ...newBlocks]);
    }

    e.target.value = "";
  };

  // Drag and drop handlers
  useEffect(() => {
    if (!dropRef.current) return;

    // Global drag events with counter for robust dragOver state
    const handleWindowDragEnter = (e: DragEvent) => {
      if (e.dataTransfer?.types?.includes("Files")) {
        dragCounter.current += 1;
        setDragOver(true);
      }
    };
    const handleWindowDragLeave = (e: DragEvent) => {
      if (e.dataTransfer?.types?.includes("Files")) {
        dragCounter.current -= 1;
        if (dragCounter.current <= 0) {
          setDragOver(false);
          dragCounter.current = 0;
        }
      }
    };
    const handleWindowDrop = async (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      dragCounter.current = 0;
      setDragOver(false);

      if (!e.dataTransfer) return;

      const files = Array.from(e.dataTransfer.files);
      const validation = validateFiles(files, contentBlocks);
      showFileValidationErrors(validation, false);

      const newBlocks = await Promise.all(
        validation.uniqueFiles.map(async (file) => {
          const block = await fileToContentBlock(file);
          if (file.type.startsWith("image/")) {
            // Generate a unique ID for this file upload
            const fileId = `${file.name}-${file.size}-${file.lastModified}`;

            // Mark this file as uploading
            setUploadingFiles(prev => ({ ...prev, [fileId]: true }));

            try {
              const formData = new FormData();
              formData.append("file", file);
              const url = await uploadImageAction(formData);
              block.metadata = { ...block.metadata, url, uploading: false, fileId };
            } catch (error) {
              console.error("Failed to upload image to Supabase:", error);
              block.metadata = { ...block.metadata, uploading: false, fileId };
            } finally {
              // Mark this file as no longer uploading
              setUploadingFiles(prev => {
                const newState = { ...prev };
                delete newState[fileId];
                return newState;
              });
            }
          }
          return block;
        }),
      );

      if (newBlocks.length > 0) {
        setContentBlocks((prev) => [...prev, ...newBlocks]);
      }
    };
    const handleWindowDragEnd = (e: DragEvent) => {
      dragCounter.current = 0;
      setDragOver(false);
    };
    window.addEventListener("dragenter", handleWindowDragEnter);
    window.addEventListener("dragleave", handleWindowDragLeave);
    window.addEventListener("drop", handleWindowDrop);
    window.addEventListener("dragend", handleWindowDragEnd);

    // Prevent default browser behavior for dragover globally
    const handleWindowDragOver = (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
    };
    window.addEventListener("dragover", handleWindowDragOver);

    // Remove element-specific drop event (handled globally)
    const handleDragOver = (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragOver(true);
    };
    const handleDragEnter = (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragOver(true);
    };
    const handleDragLeave = (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragOver(false);
    };
    const element = dropRef.current;
    element.addEventListener("dragover", handleDragOver);
    element.addEventListener("dragenter", handleDragEnter);
    element.addEventListener("dragleave", handleDragLeave);

    return () => {
      element.removeEventListener("dragover", handleDragOver);
      element.removeEventListener("dragenter", handleDragEnter);
      element.removeEventListener("dragleave", handleDragLeave);
      window.removeEventListener("dragenter", handleWindowDragEnter);
      window.removeEventListener("dragleave", handleWindowDragLeave);
      window.removeEventListener("drop", handleWindowDrop);
      window.removeEventListener("dragend", handleWindowDragEnd);
      window.removeEventListener("dragover", handleWindowDragOver);
      dragCounter.current = 0;
    };
  }, [contentBlocks]);

  const removeBlock = (idx: number) => {
    setContentBlocks((prev) => prev.filter((_, i) => i !== idx));
  };

  const resetBlocks = () => setContentBlocks([]);

  /**
   * Handle paste event for files (images, PDFs)
   * Can be used as onPaste={handlePaste} on a textarea or input
   */
  const handlePaste = async (
    e: React.ClipboardEvent<HTMLTextAreaElement | HTMLInputElement>,
  ) => {
    const items = e.clipboardData.items;
    if (!items) return;

    const files: File[] = [];
    for (let i = 0; i < items.length; i += 1) {
      const item = items[i];
      if (item.kind === "file") {
        const file = item.getAsFile();
        if (file) files.push(file);
      }
    }

    if (files.length === 0) {
      return;
    }

    e.preventDefault();

    const validation = validateFiles(files, contentBlocks);
    showFileValidationErrors(validation, true);

    const newBlocks = await Promise.all(
      validation.uniqueFiles.map(async (file) => {
        const block = await fileToContentBlock(file);
        if (file.type.startsWith("image/")) {
          // Generate a unique ID for this file upload
          const fileId = `${file.name}-${file.size}-${file.lastModified}`;

          // Mark this file as uploading
          setUploadingFiles(prev => ({ ...prev, [fileId]: true }));

          try {
            const formData = new FormData();
            formData.append("file", file);
            const url = await uploadImageAction(formData);
            block.metadata = { ...block.metadata, url, uploading: false, fileId };
          } catch (error) {
            console.error("Failed to upload image to Supabase:", error);
            block.metadata = { ...block.metadata, uploading: false, fileId };
          } finally {
            // Mark this file as no longer uploading
            setUploadingFiles(prev => {
              const newState = { ...prev };
              delete newState[fileId];
              return newState;
            });
          }
        }
        return block;
      }),
    );

    if (newBlocks.length > 0) {
      setContentBlocks((prev) => [...prev, ...newBlocks]);
    }
  };

  return {
    contentBlocks,
    setContentBlocks,
    handleFileUpload,
    dropRef,
    removeBlock,
    resetBlocks,
    dragOver,
    handlePaste,
    uploadingFiles,
  };
}
