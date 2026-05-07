import { AIMessage, ToolMessage } from "@langchain/langgraph-sdk";
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, ChevronUp } from "lucide-react";
import { useSettings } from "@/hooks/useSettings";

function isComplexValue(value: unknown): boolean {
  return Array.isArray(value) || (typeof value === "object" && value !== null);
}

type DocumentResult = {
  page_content: string;
  type?: string;
  metadata?: Record<string, unknown>;
};

function isDocumentList(value: unknown): value is DocumentResult[] {
  return (
    Array.isArray(value) &&
    value.length > 0 &&
    value.every(
      (item) =>
        typeof item === "object" &&
        item !== null &&
        "page_content" in item
    )
  );
}

function DocumentResults({ docs }: { docs: DocumentResult[] }) {
  return (
    <div className="flex flex-col gap-3">
      {docs.map((doc, idx) => {
        const meta = doc.metadata ?? {};
        const title =
          (meta.title as string) ||
          (meta.disease as string) ||
          (meta.plant as string) ||
          null;
        const url = (meta.url as string) || null;
        const source = (meta.source as string) || null;
        const relevanceScore = meta.relevance_score != null
          ? Number(meta.relevance_score).toFixed(3)
          : null;

        return (
          <div
            key={idx}
            className="rounded-lg border border-border/40 bg-muted/10 overflow-hidden"
          >
            <div className="flex items-center justify-between gap-2 border-b border-border/30 bg-muted/20 px-4 py-2">
              <div className="flex items-center gap-2 min-w-0">
                <span className="text-xs font-semibold text-foreground/50 shrink-0">
                  #{idx + 1}
                </span>
                {title ? (
                  <span className="text-xs font-medium text-foreground/80 truncate">
                    {title}
                  </span>
                ) : null}
              </div>
              <div className="flex items-center gap-2 shrink-0">
                {relevanceScore && (
                  <span className="text-xs text-muted-foreground">
                    skor: {relevanceScore}
                  </span>
                )}
                {source && (
                  <code className="rounded bg-muted/70 px-1.5 py-0.5 text-xs font-mono text-muted-foreground/80 border border-border/30">
                    {source}
                  </code>
                )}
              </div>
            </div>
            <div className="px-4 py-3">
              <p className="text-sm text-foreground/80 leading-relaxed whitespace-pre-wrap">
                {doc.page_content}
              </p>
              {url && (
                <a
                  href={url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-2 inline-block text-xs text-blue-500 hover:underline truncate max-w-full"
                >
                  {url}
                </a>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function isPlantDiseaseText(text: string): boolean {
  return (
    text.startsWith("Region-Based Classification") ||
    text.startsWith("Full-Image Classification")
  );
}

type RegionResult = {
  idx: string;
  box: string;
  label: string;
  confidence: string;
  topImageUrl: string | null;
};

type TopKResult = {
  label: string;
  score: string;
  caption: string | null;
  imageUrl: string | null;
};

function PlantDiseaseResult({ text }: { text: string }) {
  const isRegion = text.startsWith("Region-Based Classification");

  if (isRegion) {
    const methodMatch = text.match(/Region-Based Classification \(([^)]+)\)/);
    const method = methodMatch ? methodMatch[1] : "";
    const analyzedMatch = text.match(/Analyzed (\d+) detected regions/);
    const regionCount = analyzedMatch ? analyzedMatch[1] : "?";
    const filterMatch = text.match(/^Filter: (.+)$/m);
    const filter = filterMatch ? filterMatch[1].trim() : null;

    const regions: RegionResult[] = [];
    const regionBlocks = text.split(/\n(?=Region \d+)/);
    for (const block of regionBlocks) {
      const headerMatch = block.match(/^Region (\d+) \[([^\]]+)\]:/);
      if (!headerMatch) continue;
      const labelMatch = block.match(/Label: (.+) \(confidence: ([\d.]+)\)/);
      if (!labelMatch) continue;
      const urlMatch = block.match(/1\. .+? \([\d.]+\) - (https?:\/\/[^\n\s]+)/);
      regions.push({
        idx: headerMatch[1],
        box: headerMatch[2],
        label: labelMatch[1],
        confidence: labelMatch[2],
        topImageUrl: urlMatch ? urlMatch[1] : null,
      });
    }

    const imagesWithUrl = regions.filter((r) => r.topImageUrl).slice(0, 5);

    return (
      <div className="space-y-3">
        <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
          <span>
            Metode:{" "}
            <code className="font-mono bg-muted/40 px-1 rounded">{method}</code>
          </span>
          <span>·</span>
          <span>{regionCount} region terdeteksi</span>
          {filter && (
            <>
              <span>·</span>
              <span>Filter: {filter}</span>
            </>
          )}
        </div>
        {imagesWithUrl.length > 0 && (
          <div>
            <p className="text-xs text-muted-foreground mb-2">Referensi visual (prediksi teratas per region)</p>
            <div className="flex flex-wrap gap-2">
              {imagesWithUrl.map((r) => (
                <div key={r.idx} className="flex flex-col gap-1 items-center">
                  <img
                    src={r.topImageUrl!}
                    alt={r.label}
                    className="h-20 w-20 rounded-lg object-cover border border-border/30"
                  />
                  <span className="text-xs text-muted-foreground">Region {r.idx}</span>
                </div>
              ))}
            </div>
          </div>
        )}
        <div className="space-y-1.5">
          {regions.map((r) => (
            <div key={r.idx} className="rounded-lg border border-border/40 bg-muted/10 px-3 py-2">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2 min-w-0">
                  <span className="text-xs font-semibold text-foreground/50 shrink-0">Region {r.idx}</span>
                  <span className="text-sm font-medium truncate">{r.label}</span>
                </div>
                <span className="text-xs text-muted-foreground shrink-0">
                  {(parseFloat(r.confidence) * 100).toFixed(1)}%
                </span>
              </div>
              <code className="text-xs font-mono text-muted-foreground/60 mt-0.5 block">[{r.box}]</code>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // Full-image classification
  const methodMatch = text.match(/Full-Image Classification \(([^)]+)\)/);
  const method = methodMatch ? methodMatch[1] : "";
  const labelMatch = text.match(/^Predicted Label: (.+)$/m);
  const confidenceMatch = text.match(/^Confidence: ([\d.]+)$/m);
  const filterMatch = text.match(/^Filter: (.+)$/m);

  const label = labelMatch ? labelMatch[1].trim() : null;
  const confidence = confidenceMatch ? confidenceMatch[1] : null;
  const filter = filterMatch ? filterMatch[1].trim() : null;

  const topResults: TopKResult[] = [];
  const topKSection = text.indexOf("Top-");
  if (topKSection !== -1) {
    const topKText = text.slice(topKSection);
    const entryPattern =
      /\d+\. (.+?) \(([\d.]+)\)\n\s+Plant: [^\n]+\n\s+Caption: ([^\n]+)\n\s+Image: (https?:\/\/[^\n\s]+)/g;
    let m: RegExpExecArray | null;
    while ((m = entryPattern.exec(topKText)) !== null) {
      topResults.push({
        label: m[1].trim(),
        score: m[2],
        caption: m[3].trim(),
        imageUrl: m[4],
      });
    }
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
        <span>
          Metode:{" "}
          <code className="font-mono bg-muted/40 px-1 rounded">{method}</code>
        </span>
        {filter && (
          <>
            <span>·</span>
            <span>Filter: {filter}</span>
          </>
        )}
      </div>
      {label && (
        <div className="rounded-lg border border-border/40 bg-muted/10 px-4 py-2.5">
          <div className="text-xs text-muted-foreground mb-1">Prediksi utama</div>
          <div className="flex items-center justify-between gap-2">
            <span className="text-sm font-medium">{label}</span>
            {confidence && (
              <span className="text-xs text-muted-foreground">
                kepercayaan: {(parseFloat(confidence) * 100).toFixed(1)}%
              </span>
            )}
          </div>
        </div>
      )}
      {topResults.length > 0 && (
        <>
          <div>
            <p className="text-xs text-muted-foreground mb-2">Referensi visual teratas</p>
            <div className="flex flex-wrap gap-2">
              {topResults
                .filter((r) => r.imageUrl)
                .slice(0, 5)
                .map((r, idx) => (
                  <div key={idx} className="flex flex-col gap-1 items-center">
                    <img
                      src={r.imageUrl!}
                      alt={r.label}
                      className="h-20 w-20 rounded-lg object-cover border border-border/30"
                    />
                    <span className="text-xs text-muted-foreground text-center">
                      #{idx + 1} · {(parseFloat(r.score) * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
            </div>
          </div>
          <div className="space-y-1.5">
            {topResults.map((r, idx) => (
              <div key={idx} className="rounded-lg border border-border/40 bg-muted/10 px-3 py-2">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="text-xs font-semibold text-foreground/50 shrink-0">#{idx + 1}</span>
                    <span className="text-sm font-medium truncate">{r.label}</span>
                  </div>
                  <span className="text-xs text-muted-foreground shrink-0">
                    {(parseFloat(r.score) * 100).toFixed(1)}%
                  </span>
                </div>
                {r.caption && (
                  <p className="mt-1 text-xs text-muted-foreground line-clamp-2">{r.caption}</p>
                )}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

export function ToolCalls({
  toolCalls,
  isLoading,
}: {
  toolCalls: AIMessage["tool_calls"];
  isLoading?: boolean;
}) {
  const { userSettings } = useSettings();
  if (!toolCalls || toolCalls.length === 0) return null;

  return (
    <div className={`w-full grid ${userSettings.chatWidth === "default" ? "max-w-3xl" : "max-w-5xl"} grid-rows-[1fr_auto] gap-4`}>
      {toolCalls.map((tc, idx) => {
        return <ToolCallItem key={idx} toolCall={tc} isLoading={isLoading} />;
      })}
    </div>
  );
}

function ToolCallItem({
  toolCall,
  isLoading
}: {
  toolCall: NonNullable<AIMessage["tool_calls"]>[number];
  isLoading?: boolean;
}) {
  const { userSettings } = useSettings();
  const [isExpanded, setIsExpanded] = useState(true);

  useEffect(() => {
    if (userSettings.autoCollapseToolCalls && isLoading === false) {
      setIsExpanded(false);
    }
  }, [isLoading, userSettings.autoCollapseToolCalls]);

  const rawArgs = toolCall.args as unknown;
  let args: Record<string, unknown> = {};
  if (rawArgs && typeof rawArgs === "object" && !Array.isArray(rawArgs)) {
    args = rawArgs as Record<string, unknown>;
  } else if (typeof rawArgs === "string" && rawArgs.trim().length > 0) {
    try {
      const parsed = JSON.parse(rawArgs);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        args = parsed as Record<string, unknown>;
      }
    } catch {
      args = { input: rawArgs };
    }
  }
  const hasArgs = Object.keys(args).length > 0;
  const argEntries = Object.entries(args);

  return (
    <div className="overflow-hidden rounded-xl border border-border/50 dark:border-border bg-card shadow-sm transition-all duration-200 hover:shadow-md hover:border-border dark:hover:border-border/80">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full border-b border-border/50 dark:border-border bg-muted/30 dark:bg-muted/50 px-5 py-3.5 text-left transition-all duration-200 hover:bg-muted/50 dark:hover:bg-muted/70"
      >
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="flex h-7 w-7 items-center justify-center rounded-full bg-foreground/8 ring-1 ring-foreground/5">
              <svg
                className="h-3.5 w-3.5 text-foreground/70"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M13 10V3L4 14h7v7l9-11h-7z"
                />
              </svg>
            </div>
            <h3 className="font-medium text-foreground text-sm">
              {toolCall.name}
              {toolCall.id && (
                <code className="ml-2 rounded-md bg-muted/70 px-2 py-0.5 text-xs font-mono text-muted-foreground/80 border border-border/30">
                  {toolCall.id.slice(0, 8)}...
                </code>
              )}
            </h3>
          </div>
          <motion.div
            animate={{ rotate: isExpanded ? 0 : -90 }}
            transition={{ duration: 0.25, ease: "easeInOut" }}
          >
            <ChevronDown className="h-4 w-4 text-muted-foreground/70" />
          </motion.div>
        </div>
      </button>
      <AnimatePresence initial={false}>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: [0.4, 0, 0.2, 1] }}
            className="overflow-hidden"
          >
            {hasArgs ? (
              <div className="bg-card">
                <table className="min-w-full">
                  <tbody className="divide-y divide-border/40">
                    {argEntries.map(([key, value], argIdx) => (
                      <tr
                        key={argIdx}
                        className="transition-colors duration-150 hover:bg-muted/30"
                      >
                        <td className="px-5 py-3 text-xs font-semibold whitespace-nowrap text-foreground/70 bg-muted/20 w-1/4">
                          {key}
                        </td>
                        <td className="px-5 py-3 text-sm text-foreground/85">
                          {isComplexValue(value) ? (
                            <code className="block rounded-lg bg-muted/40 px-3 py-2 font-mono text-xs break-all border border-border/30">
                              {JSON.stringify(value, null, 2)}
                            </code>
                          ) : (
                            <span className="font-normal">{String(value)}</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="bg-card px-5 py-4">
                <code className="text-xs text-muted-foreground/60 italic">
                  Tidak ada argumen
                </code>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export function ToolResult({
  message,
  isLoading
}: {
  message: ToolMessage;
  isLoading?: boolean;
}) {
  const { userSettings } = useSettings();
  const [isExpanded, setIsExpanded] = useState(true);

  useEffect(() => {
    if (userSettings.autoCollapseToolCalls && isLoading === false) {
      setIsExpanded(false);
    }
  }, [isLoading, userSettings.autoCollapseToolCalls]);

  const isArrayContent = Array.isArray(message.content);
  let parsedContent: unknown;
  let isJsonContent = false;
  let imageUrls: string[] = [];

  if (isArrayContent) {
    const blocks = message.content as Array<Record<string, unknown>>;
    const textParts: string[] = [];
    for (const block of blocks) {
      if (block.type === "text" && typeof block.text === "string") {
        textParts.push(block.text);
      } else if (block.type === "image_url") {
        const url =
          typeof block.image_url === "string"
            ? block.image_url
            : (block.image_url as Record<string, unknown>)?.url as string;
        if (url) imageUrls.push(url);
      }
    }
    parsedContent = textParts.join("\n");
  } else {
    try {
      if (typeof message.content === "string") {
        parsedContent = JSON.parse(message.content);
        isJsonContent = isComplexValue(parsedContent);
      }
    } catch {
      parsedContent = message.content;
    }
  }

  const contentStr = isJsonContent
    ? JSON.stringify(parsedContent, null, 2)
    : isArrayContent
    ? String(parsedContent ?? "")
    : String(message.content);
  const contentLines = contentStr.split("\n");
  const shouldTruncate = contentLines.length > 4 || contentStr.length > 500;
  const displayedContent =
    shouldTruncate && !isExpanded
      ? contentStr.length > 500
        ? contentStr.slice(0, 500) + "..."
        : contentLines.slice(0, 4).join("\n") + "\n..."
      : contentStr;

  return (
    <div className={`w-full grid ${userSettings.chatWidth === "default" ? "max-w-3xl" : "max-w-5xl"} grid-rows-[1fr_auto] gap-0`}>
      <div className="overflow-hidden rounded-xl border border-border/50 dark:border-border bg-card shadow-sm transition-all duration-200 hover:shadow-md hover:border-border dark:hover:border-border/80">
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="w-full border-b border-border/50 dark:border-border bg-muted/30 dark:bg-muted/50 px-5 py-3.5 text-left transition-all duration-200 hover:bg-muted/50 dark:hover:bg-muted/70 cursor-pointer"
        >
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <div className="flex h-7 w-7 items-center justify-center rounded-full bg-foreground/8 ring-1 ring-foreground/5">
                <svg
                  className="h-3.5 w-3.5 text-foreground/70"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M5 13l4 4L19 7"
                  />
                </svg>
              </div>
              {message.name ? (
                <h3 className="font-medium text-foreground text-sm flex gap-2 items-center">
                  Hasil Alat
                  <code className="rounded-md bg-muted/70 px-2 py-0.5 text-xs font-mono text-muted-foreground/80 border border-border/30">
                    {message.name}
                  </code>
                </h3>
              ) : (
                <h3 className="font-medium text-foreground text-sm">
                  Hasil Alat
                </h3>
              )}
            </div>
            <div className="flex items-center gap-2">
              {message.tool_call_id && (
                <code className="rounded-md bg-muted/70 px-2 py-0.5 text-xs font-mono text-muted-foreground/80 border border-border/30">
                  {message.tool_call_id.slice(0, 8)}...
                </code>
              )}
              <motion.div
                animate={{ rotate: isExpanded ? 0 : -90 }}
                transition={{ duration: 0.25, ease: "easeInOut" }}
              >
                <ChevronDown className="h-4 w-4 text-muted-foreground/70" />
              </motion.div>
            </div>
          </div>
        </button>
        <AnimatePresence initial={false}>
          {isExpanded && (
            <motion.div
              className="min-w-full bg-card overflow-hidden"
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.25, ease: [0.4, 0, 0.2, 1] }}
            >
              <div className="p-3">
                <AnimatePresence
                  mode="wait"
                  initial={false}
                >
                  <motion.div
                    key={isExpanded ? "expanded" : "collapsed"}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ duration: 0.2, ease: [0.4, 0, 0.2, 1] }}
                  >
                    {imageUrls.length > 0 && (
                      <div className="mb-3 flex flex-wrap gap-2">
                        {imageUrls.map((url, idx) => (
                          <img
                            key={idx}
                            src={url}
                            alt={`Hasil alat gambar ${idx + 1}`}
                            className="max-h-60 rounded-lg object-contain border border-border/30"
                          />
                        ))}
                      </div>
                    )}
                    {isDocumentList(parsedContent) ? (
                      <DocumentResults docs={isExpanded ? parsedContent : parsedContent.slice(0, 3)} />
                    ) : isPlantDiseaseText(contentStr) ? (
                      <PlantDiseaseResult text={contentStr} />
                    ) : isJsonContent ? (
                      <table className="min-w-full">
                        <tbody className="divide-y divide-border/40">
                          {(Array.isArray(parsedContent)
                            ? isExpanded
                              ? parsedContent
                              : parsedContent.slice(0, 5)
                            : Object.entries(parsedContent as Record<string, unknown>)
                          ).map((item, argIdx) => {
                            const [key, value] = Array.isArray(parsedContent)
                              ? [argIdx, item]
                              : [item[0], item[1]];
                            return (
                              <tr
                                key={argIdx}
                                className="transition-colors duration-150 hover:bg-muted/30"
                              >
                                <td className="px-5 py-3 text-xs font-semibold whitespace-nowrap text-foreground/70 bg-muted/20 w-1/4">
                                  {key}
                                </td>
                                <td className="px-5 py-3 text-sm text-foreground/85">
                                  {isComplexValue(value) ? (
                                    <code className="block rounded-lg bg-muted/40 px-3 py-2 font-mono text-xs break-all border border-border/30">
                                      {JSON.stringify(value, null, 2)}
                                    </code>
                                  ) : (
                                    <span className="font-normal">{String(value)}</span>
                                  )}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    ) : (
                      <code className="block rounded-lg bg-muted/40 px-3 py-2.5 text-sm font-mono border border-border/30 leading-relaxed whitespace-pre-wrap">
                        {displayedContent}
                      </code>
                    )}
                  </motion.div>
                </AnimatePresence>
              </div>
              {((shouldTruncate && !isJsonContent && !isDocumentList(parsedContent) && !isPlantDiseaseText(contentStr)) ||
                (isJsonContent &&
                  !isDocumentList(parsedContent) &&
                  Array.isArray(parsedContent) &&
                  parsedContent.length > 5) ||
                (isDocumentList(parsedContent) && parsedContent.length > 3)) && (
                <motion.button
                  onClick={(e) => {
                    e.stopPropagation();
                    setIsExpanded(!isExpanded);
                  }}
                  className="flex w-full cursor-pointer items-center justify-center gap-2 border-t border-border/40 bg-muted/20 py-2.5 text-xs font-medium text-foreground/70 transition-all duration-150 ease-in-out hover:bg-muted/40"
                  initial={{ scale: 1 }}
                  whileHover={{ scale: 1.002 }}
                  whileTap={{ scale: 0.998 }}
                >
                  {isExpanded ? (
                    <>
                      <ChevronUp className="h-3.5 w-3.5" />
                      <span>Tampilkan lebih sedikit</span>
                    </>
                  ) : (
                    <>
                      <ChevronDown className="h-3.5 w-3.5" />
                      <span>Tampilkan lebih banyak</span>
                    </>
                  )}
                </motion.button>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
