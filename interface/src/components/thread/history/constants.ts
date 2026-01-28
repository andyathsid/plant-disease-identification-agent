// Sidebar dimensions
export const SIDEBAR_WIDTH = 300;

// Thread display settings
export const MAX_THREAD_TITLE_LENGTH = 16;
export const SKELETON_LOADING_COUNT = 30;

// Styling constants
export const SCROLLBAR_STYLES =
  "[&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-border [&::-webkit-scrollbar-track]:bg-transparent";

export const THREAD_ITEM_PADDING = "px-3 py-2";
export const ICON_SIZE_SM = "size-5";
export const BUTTON_SIZE_SM = "h-7 w-7";

// UI Text (Indonesian)
export const UI_TEXT = {
  newChat: "Obrolan Baru",
  rename: "Ubah Nama",
  delete: "Hapus",
  deleteConfirm: "Apakah Anda yakin ingin menghapus percakapan ini?",
  deleteSuccess: "Percakapan berhasil dihapus",
  deleteError: "Gagal menghapus percakapan",
  updateSuccess: "Judul percakapan diperbarui",
  updateError: "Gagal memperbarui judul",
} as const;
