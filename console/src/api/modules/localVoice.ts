import { request } from "../request";

export type LocalVoiceAsset =
  | "zipformer"
  | "kws"
  | "kokoro"
  | "kokoro_int8"
  | "qwen3";
export type LocalVoiceTest = "wake_word" | "asr" | "tts";

export interface LocalVoiceModelStatus {
  id: LocalVoiceAsset;
  name: string;
  path: string;
  installed: boolean;
  complete: boolean;
  size_bytes: number;
  missing_files: string[];
  download_url: string;
}

export interface LocalVoiceModelsResponse {
  models: LocalVoiceModelStatus[];
  missing_dependencies: string[];
}

export interface LocalVoiceDownloadStatus {
  status: "idle" | "downloading" | "completed" | "failed";
  asset: LocalVoiceAsset | null;
  downloaded_bytes: number;
  total_bytes: number | null;
  error: string | null;
}

export interface LocalVoiceTestStatus {
  status: "idle" | "running" | "passed" | "failed" | "timed_out";
  test: LocalVoiceTest | null;
  message: string | null;
  transcript: string | null;
  keyword: string | null;
}

export interface LocalVoiceDependencyInstallStatus {
  status: "idle" | "installing" | "completed" | "failed";
  packages: string[];
  error: string | null;
}

export interface LocalVoiceDependencyItem {
  id: string;
  name: string;
  spec: string;
  description: string;
  installed: boolean;
  required: boolean;
  can_uninstall: boolean;
}

export interface LocalVoiceDependencyStatus {
  items: LocalVoiceDependencyItem[];
  install: LocalVoiceDependencyInstallStatus;
}

export interface LocalVoiceDeviceOption {
  value: string;
  label: string;
  available: boolean;
  description: string;
}

export interface Qwen3VoiceCloneResponse {
  voice_id: string;
  message: string;
}

export const localVoiceApi = {
  getModelStatus: (config: Record<string, unknown>) =>
    request<LocalVoiceModelsResponse>("/local-voice/models/status", {
      method: "POST",
      body: JSON.stringify({ config }),
    }),
  startModelDownload: (
    asset: LocalVoiceAsset,
    config: Record<string, unknown>,
  ) =>
    request<{ status: "accepted"; message: string }>(
      "/local-voice/models/download",
      {
        method: "POST",
        body: JSON.stringify({ asset, config }),
      },
    ),
  deleteModel: (asset: LocalVoiceAsset, config: Record<string, unknown>) =>
    request<{ status: "accepted"; message: string }>(
      "/local-voice/models/delete",
      {
        method: "POST",
        body: JSON.stringify({ asset, config }),
      },
    ),
  getModelDownloadStatus: () =>
    request<LocalVoiceDownloadStatus>("/local-voice/models/download"),
  installDependencies: (
    packages: string[] = [],
    config: Record<string, unknown> = {},
  ) =>
    request<{ status: "accepted"; message: string }>(
      "/local-voice/dependencies/install",
      { method: "POST", body: JSON.stringify({ packages, config }) },
    ),
  getDependencyInstallStatus: () =>
    request<LocalVoiceDependencyInstallStatus>(
      "/local-voice/dependencies/install",
    ),
  getDependencyStatus: (config: Record<string, unknown>) =>
    request<LocalVoiceDependencyStatus>("/local-voice/dependencies/status", {
      method: "POST",
      body: JSON.stringify({ config }),
    }),
  uninstallDependencies: (packages: string[]) =>
    request<{ status: "accepted"; message: string }>(
      "/local-voice/dependencies/uninstall",
      { method: "POST", body: JSON.stringify({ packages }) },
    ),
  getQwen3Devices: () =>
    request<LocalVoiceDeviceOption[]>("/local-voice/qwen3/devices"),
  startTest: (test: LocalVoiceTest, config: Record<string, unknown>) =>
    request<{ status: "accepted"; message: string }>("/local-voice/tests", {
      method: "POST",
      body: JSON.stringify({ test, config }),
    }),
  getTestStatus: () => request<LocalVoiceTestStatus>("/local-voice/tests"),
  cloneQwen3Voice: (config: Record<string, unknown>) =>
    request<Qwen3VoiceCloneResponse>("/local-voice/qwen3/voice-clone", {
      method: "POST",
      body: JSON.stringify({ config }),
    }),
};
