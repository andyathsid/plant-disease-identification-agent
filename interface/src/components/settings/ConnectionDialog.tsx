"use client";

import { useState } from "react";
import { useQueryState } from "nuqs";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { PasswordInput } from "@/components/ui/password-input";
import { ArrowRight } from "lucide-react";
import { getApiKey } from "@/lib/api-key";

// Default values for the form
const DEFAULT_API_URL = "http://localhost:2024";

interface ConnectionDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function ConnectionDialog({ open, onOpenChange }: ConnectionDialogProps) {
  const [apiUrl, setApiUrl] = useQueryState("apiUrl");
  const [apiKey, _setApiKey] = useState(() => getApiKey() || "");

  const setApiKey = (key: string) => {
    window.localStorage.setItem("lg:chat:apiKey", key);
    _setApiKey(key);
  };

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    const form = e.target as HTMLFormElement;
    const formData = new FormData(form);
    const newApiUrl = formData.get("apiUrl") as string;
    const newApiKey = formData.get("apiKey") as string;

    setApiUrl(newApiUrl);
    setApiKey(newApiKey);

    form.reset();
    onOpenChange(false);

    // Reload the page to apply new connection settings
    window.location.reload();
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle className="text-xl font-semibold">Koneksi Baru</DialogTitle>
          <DialogDescription>
            Konfigurasi endpoint deployment LangGraph baru
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="flex flex-col gap-6 pt-4">
          <div className="flex flex-col gap-2">
            <Label htmlFor="apiUrl">
              URL Deployment<span className="text-rose-500">*</span>
            </Label>
            <p className="text-muted-foreground text-sm">
              Ini adalah URL deployment LangGraph Anda. Bisa berupa deployment lokal atau produksi.
            </p>
            <Input
              id="apiUrl"
              name="apiUrl"
              className="bg-background"
              defaultValue={apiUrl || DEFAULT_API_URL}
              required
            />
          </div>

          <div className="flex flex-col gap-2">
            <Label htmlFor="apiKey">LangSmith API Key</Label>
            <p className="text-muted-foreground text-sm">
              Ini <strong>TIDAK</strong> diperlukan jika menggunakan server LangGraph lokal. Nilai ini disimpan di penyimpanan lokal browser Anda dan hanya digunakan untuk mengautentikasi permintaan yang dikirim ke server LangGraph Anda.
            </p>
            <PasswordInput
              id="apiKey"
              name="apiKey"
              defaultValue={apiKey ?? ""}
              className="bg-background"
              placeholder="lsv2_pt_..."
            />
          </div>

          <div className="mt-2 flex justify-end gap-2">
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
            >
              Batal
            </Button>
            <Button type="submit" size="default">
              Hubungkan
              <ArrowRight className="ml-2 size-4" />
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
