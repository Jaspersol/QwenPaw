import { useCallback, useEffect, useRef, useState } from "react";
import type { FormInstance } from "antd";
import { Alert, Input, Progress, Space, Tag, Tooltip, Typography } from "antd";
import { Button, Form } from "@agentscope-ai/design";
import {
  localVoiceApi,
  type LocalVoiceAsset,
  type LocalVoiceDependencyInstallStatus,
  type LocalVoiceDependencyItem,
  type LocalVoiceDownloadStatus,
  type LocalVoiceModelStatus,
  type LocalVoiceTest,
  type LocalVoiceTestStatus,
} from "@/api/modules/localVoice";
import { useAppMessage } from "@/hooks/useAppMessage";
import { useQwen3VoiceClone } from "@/hooks/useQwen3VoiceClone";

interface LocalVoiceToolsProps {
  form: FormInstance<Record<string, unknown>>;
}

const assetLabels: Record<LocalVoiceAsset, string> = {
  zipformer: "ASR · Zipformer",
  kws: "Wake word · KWS",
  kokoro: "TTS · Kokoro v1.1",
  kokoro_int8: "TTS · Kokoro v1.1 (int8)",
  qwen3: "TTS · Qwen3-TTS 0.6B (HF)",
};
const assetDescriptions: Record<LocalVoiceAsset, string> = {
  zipformer:
    "Streaming bilingual (zh/en) ASR model used by the local Zipformer provider.",
  kws: "Always-on keyword spotting model that gates ASR until the wake word is detected.",
  kokoro:
    "Offline bilingual TTS model used by the Kokoro provider (balanced quality).",
  kokoro_int8:
    "Quantized Kokoro model with lower CPU usage and slightly reduced quality.",
  qwen3:
    "Hugging Face snapshot of Qwen/Qwen3-TTS-12Hz-0.6B-Base used by the local Qwen3-TTS backend.",
};
function testLabel(
  kind: LocalVoiceTest,
  sttProvider: string,
  ttsProvider: string,
): string {
  if (kind === "wake_word") return "Test wake word (12 s)";
  if (kind === "asr") {
    if (sttProvider === "openai") return "Test ASR via API (10 s)";
    if (sttProvider === "aliyun") return "Test ASR via DashScope (10 s)";
    return "Test ASR (10 s)";
  }
  if (ttsProvider === "openai") return "Play TTS via API";
  if (ttsProvider === "aliyun") return "Play TTS via DashScope";
  return "Play TTS test";
}

function providerLabel(provider: string): string {
  if (provider === "openai") return "OpenAI-compatible API";
  if (provider === "aliyun") return "Aliyun DashScope";
  if (provider === "qwen3") return "Qwen3-TTS 0.6B";
  if (provider === "sherpa_zipformer") return "local Zipformer";
  if (provider === "kokoro") return "local Kokoro";
  return provider;
}

function testHint(
  kind: LocalVoiceTest,
  sttProvider: string,
  ttsProvider: string,
): string {
  if (kind === "wake_word") {
    return "Listens for the configured KWS wake word and reports the detected keyword.";
  }
  if (kind === "asr") {
    return `Records 10 seconds from the selected microphone and transcribes it with ${providerLabel(
      sttProvider,
    )}.`;
  }
  return `Synthesizes a short test phrase with ${providerLabel(
    ttsProvider,
  )} and plays it on the selected output device.`;
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  const units = ["KB", "MB", "GB"];
  let size = value / 1024;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${size.toFixed(1)} ${units[index]}`;
}

export function LocalVoiceTools({ form }: LocalVoiceToolsProps) {
  const { message } = useAppMessage();
  const [models, setModels] = useState<LocalVoiceModelStatus[]>([]);
  const [dependencyItems, setDependencyItems] = useState<
    LocalVoiceDependencyItem[]
  >([]);
  const [download, setDownload] = useState<LocalVoiceDownloadStatus | null>(
    null,
  );
  const [dependencyInstall, setDependencyInstall] =
    useState<LocalVoiceDependencyInstallStatus | null>(null);
  const [test, setTest] = useState<LocalVoiceTestStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [deletingAsset, setDeletingAsset] = useState<LocalVoiceAsset | null>(
    null,
  );
  const [uninstallingDependency, setUninstallingDependency] = useState<
    string | null
  >(null);
  const completedDownload = useRef(false);
  const completedDependencyInstall = useRef(false);
  const [pollNonce, setPollNonce] = useState(0);
  const values = useCallback(
    () => form.getFieldsValue(true) as Record<string, unknown>,
    [form],
  );
  const onCloned = useCallback(
    (voiceId: string) => form.setFieldValue("tts_voice", voiceId),
    [form],
  );
  const { cloning, cloneVoice } = useQwen3VoiceClone(values, onCloned);
  const sttProvider = String(
    Form.useWatch("stt_provider", form) ?? "sherpa_zipformer",
  );
  const ttsProvider = String(Form.useWatch("tts_provider", form) ?? "kokoro");
  const qwen3Backend = String(Form.useWatch("qwen3_backend", form) ?? "api");
  const qwen3Device = String(Form.useWatch("qwen3_device", form) ?? "cpu");
  const usesQwen3Api = ttsProvider === "qwen3" && qwen3Backend === "api";
  const wakeWordEnabled = Boolean(
    Form.useWatch("wake_word_enabled", form) ?? true,
  );
  const usesLocalAsr = sttProvider === "sherpa_zipformer";
  const usesLocalTts = ttsProvider === "kokoro";
  const usesQwen3Local = ttsProvider === "qwen3" && qwen3Backend === "local";
  const usesLocalModels = usesLocalAsr || usesLocalTts || usesQwen3Local;
  const visibleModels = models.filter((model) => {
    if (model.id.startsWith("kokoro")) return usesLocalTts;
    if (model.id === "qwen3") return usesQwen3Local;
    if (model.id === "kws") return usesLocalAsr && wakeWordEnabled;
    return usesLocalAsr;
  });
  const requiredMissingItems = dependencyItems.filter(
    (item) => item.required && !item.installed,
  );
  const requiredMissing = requiredMissingItems.map((item) => item.id);

  const refreshModels = useCallback(async () => {
    setLoading(true);
    try {
      const [modelsResult, dependenciesResult] = await Promise.all([
        localVoiceApi.getModelStatus(values()),
        localVoiceApi.getDependencyStatus(values()),
      ]);
      setModels(modelsResult.models);
      setDependencyItems(dependenciesResult.items);
      setDependencyInstall(dependenciesResult.install);
    } catch (error) {
      message.error(
        error instanceof Error
          ? error.message
          : "Unable to read voice model status",
      );
    } finally {
      setLoading(false);
    }
  }, [message, values]);

  useEffect(() => {
    void refreshModels();
  }, [refreshModels]);
  // Poll task status while a download / dependency install / diagnostic test
  // is actually running, then stop. Idle status endpoints would otherwise be
  // hit every second for as long as the Local Voice settings are open, which
  // floods the backend access log.
  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    const poll = async () => {
      let downloadStatus: LocalVoiceDownloadStatus;
      let dependencyStatus: LocalVoiceDependencyInstallStatus;
      let testStatus: LocalVoiceTestStatus;
      try {
        [downloadStatus, dependencyStatus, testStatus] = await Promise.all([
          localVoiceApi.getModelDownloadStatus(),
          localVoiceApi.getDependencyInstallStatus(),
          localVoiceApi.getTestStatus(),
        ]);
      } catch {
        /* API may restart while settings are open. */
        return;
      }
      if (cancelled) return;

      setDownload(downloadStatus);
      setDependencyInstall(dependencyStatus);
      setTest(testStatus);
      if (
        downloadStatus.status === "completed" &&
        !completedDownload.current
      ) {
        completedDownload.current = true;
        void refreshModels();
      }
      if (
        dependencyStatus.status === "completed" &&
        !completedDependencyInstall.current
      ) {
        completedDependencyInstall.current = true;
        void refreshModels();
      }

      const active =
        downloadStatus.status === "downloading" ||
        dependencyStatus.status === "installing" ||
        testStatus.status === "running";
      if (!active || document.visibilityState === "hidden") return;
      timer = window.setTimeout(() => void poll(), 1000);
    };

    void poll();
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [refreshModels, pollNonce]);

  const startDownload = async (asset: LocalVoiceAsset) => {
    try {
      await localVoiceApi.startModelDownload(asset, values());
      completedDownload.current = false;
      setDownload({
        status: "downloading",
        asset,
        downloaded_bytes: 0,
        total_bytes: null,
        error: null,
      });
      setPollNonce((value) => value + 1);
    } catch (error) {
      message.error(
        error instanceof Error ? error.message : "Unable to start download",
      );
    }
  };
  const deleteModel = async (asset: LocalVoiceAsset, path: string) => {
    if (
      !window.confirm(
        `Delete the local model directory?\n\n${path}\n\nThis cannot be undone.`,
      )
    ) {
      return;
    }
    setDeletingAsset(asset);
    try {
      const result = await localVoiceApi.deleteModel(asset, values());
      message.success(result.message);
      await refreshModels();
    } catch (error) {
      message.error(
        error instanceof Error ? error.message : "Unable to delete model",
      );
    } finally {
      setDeletingAsset(null);
    }
  };
  const installDependencies = async (packages: string[] = []) => {
    const selected = packages.length > 0 ? packages : requiredMissing;
    if (selected.length === 0) {
      message.info("All required dependencies are already installed");
      return;
    }
    try {
      await localVoiceApi.installDependencies(selected, values());
      completedDependencyInstall.current = false;
      setDependencyInstall({
        status: "installing",
        packages: selected,
        error: null,
      });
      setPollNonce((value) => value + 1);
    } catch (error) {
      message.error(
        error instanceof Error
          ? error.message
          : "Unable to install Local Voice dependencies",
      );
    }
  };
  const uninstallDependency = async (item: LocalVoiceDependencyItem) => {
    const requiredHint = item.required
      ? "\n\nIt is marked as required by the current settings."
      : "";
    const torchHint =
      item.id === "torch_cuda"
        ? "\n\nIf another application is using this PyTorch build, it must be restarted after removal."
        : "";
    if (
      !window.confirm(
        `Uninstall ${item.name} (${item.spec})?\n\n${item.description}${requiredHint}${torchHint}\n\nThis runs pip uninstall on the QwenPaw interpreter.`,
      )
    ) {
      return;
    }
    setUninstallingDependency(item.id);
    try {
      const result = await localVoiceApi.uninstallDependencies([item.id]);
      message.success(result.message);
      await refreshModels();
    } catch (error) {
      message.error(
        error instanceof Error
          ? error.message
          : `Unable to uninstall ${item.name}`,
      );
    } finally {
      setUninstallingDependency(null);
    }
  };
  const startTest = async (kind: LocalVoiceTest) => {
    try {
      await localVoiceApi.startTest(kind, values());
      setTest({
        status: "running",
        test: kind,
        message: null,
        transcript: null,
        keyword: null,
      });
      setPollNonce((value) => value + 1);
    } catch (error) {
      message.error(
        error instanceof Error ? error.message : "Unable to start test",
      );
    }
  };
  const progress =
    download?.status === "downloading" && download.total_bytes
      ? Math.round((download.downloaded_bytes / download.total_bytes) * 100)
      : 0;

  return (
    <Space
      direction="vertical"
      size="middle"
      style={{ width: "100%", marginTop: 8 }}
    >
      {usesLocalModels ? (
        <>
          <Typography.Title level={5} style={{ margin: 0 }}>
            Local models
          </Typography.Title>
          <Typography.Text type="secondary">
            The displayed location is the effective path after configured and
            environment fallbacks. Downloads use official sherpa-onnx release
            archives; the Qwen3-TTS model uses a Hugging Face snapshot.
          </Typography.Text>
          {visibleModels.map((model) => {
            const downloading =
              download?.status === "downloading" && download.asset === model.id;
            return (
              <div
                key={model.id}
                style={{
                  border: "1px solid #f0f0f0",
                  borderRadius: 6,
                  padding: 12,
                }}
              >
                <Space direction="vertical" size={6} style={{ width: "100%" }}>
                  <Space>
                    <Tooltip title={assetDescriptions[model.id]}>
                      <Typography.Text strong>
                        {assetLabels[model.id]}
                      </Typography.Text>
                    </Tooltip>
                    <Tag
                      color={
                        model.complete
                          ? "success"
                          : model.installed
                          ? "warning"
                          : "default"
                      }
                    >
                      {model.complete
                        ? "Ready"
                        : model.installed
                        ? "Incomplete"
                        : "Not downloaded"}
                    </Tag>
                  </Space>
                  <Typography.Text
                    type="secondary"
                    style={{ wordBreak: "break-all" }}
                  >
                    {model.path}
                  </Typography.Text>
                  {model.complete && (
                    <Typography.Text type="secondary">
                      {formatBytes(model.size_bytes)}
                    </Typography.Text>
                  )}
                  {!model.complete && model.missing_files.length > 0 && (
                    <Typography.Text type="secondary">
                      Missing: {model.missing_files.join(", ")}
                    </Typography.Text>
                  )}
                  <Tooltip
                    title={
                      model.complete
                        ? "Model is ready and no download is needed."
                        : model.installed
                        ? "The directory already exists. Remove it or choose a different directory before downloading."
                        : `Download the official ${model.name} archive to the directory shown above.`
                    }
                  >
                    <span style={{ display: "inline-flex" }}>
                      <Button
                        size="small"
                        style={{ minWidth: 120 }}
                        disabled={
                          model.complete ||
                          Boolean(download?.status === "downloading") ||
                          loading
                        }
                        loading={downloading}
                        onClick={() => void startDownload(model.id)}
                      >
                        {model.installed ? "Directory exists" : "Download"}
                      </Button>
                    </span>
                  </Tooltip>
                  {model.installed && (
                    <Tooltip title={`Delete ${model.path}`}>
                      <span style={{ display: "inline-flex" }}>
                        <Button
                          size="small"
                          danger
                          disabled={
                            Boolean(download?.status === "downloading") ||
                            loading ||
                            deletingAsset !== null
                          }
                          loading={deletingAsset === model.id}
                          onClick={() => void deleteModel(model.id, model.path)}
                        >
                          Delete
                        </Button>
                      </span>
                    </Tooltip>
                  )}
                </Space>
              </div>
            );
          })}
          <Tooltip title="Re-reads the configured model directories and updates the download status.">
            <span style={{ display: "inline-flex" }}>
              <Button
                size="small"
                onClick={() => void refreshModels()}
                loading={loading}
              >
                Refresh model status
              </Button>
            </span>
          </Tooltip>
          {download?.status === "downloading" && (
            <Progress
              percent={progress}
              status="active"
              format={() =>
                download.total_bytes
                  ? `${formatBytes(download.downloaded_bytes)} / ${formatBytes(
                      download.total_bytes,
                    )}`
                  : formatBytes(download.downloaded_bytes)
              }
            />
          )}
          {download?.status === "failed" && (
            <Alert
              type="error"
              showIcon
              message="Model download failed"
              description={download.error}
            />
          )}
        </>
      ) : (
        <Alert
          type="info"
          showIcon
          message={
            usesQwen3Local
              ? "Local Qwen3-TTS selected"
              : "API providers selected"
          }
          description={
            usesQwen3Local
              ? "The 0.6B model is loaded from qwen3_model_dir (or Hugging Face) through qwen-tts; no sherpa-onnx model download is required."
              : "Local ASR/KWS/TTS models are not required. The hardware tests below use the configured API endpoints directly."
          }
        />
      )}
      <Typography.Title level={5} style={{ margin: "8px 0 0" }}>
        Dependencies
      </Typography.Title>
      <Typography.Text type="secondary">
        All installable Local Voice dependencies are listed below. Entries
        marked &quot;Required by current settings&quot; are needed by the
        ASR/TTS providers selected above.
      </Typography.Text>
      <div
        style={{
          border: "1px solid #f0f0f0",
          borderRadius: 8,
          background: "rgba(0, 0, 0, 0.02)",
          padding: "10px 12px",
          marginBottom: 12,
        }}
      >
        <Typography.Text
          type="secondary"
          style={{ fontSize: 12, display: "block", marginBottom: 8 }}
        >
          Optional pip mirror / source overrides. Leave empty to use the
          official PyPI and PyTorch indexes.
        </Typography.Text>
        <Form.Item
          name="pip_index_url"
          label="PyPI mirror"
          tooltip="Custom pip index URL for normal Python dependencies, e.g. https://pypi.tuna.tsinghua.edu.cn/simple"
          style={{ marginBottom: 8 }}
        >
          <Input
            placeholder="https://pypi.tuna.tsinghua.edu.cn/simple"
            allowClear
          />
        </Form.Item>
        <Form.Item
          name="torch_index_url"
          label="PyTorch mirror"
          tooltip="Custom PyTorch wheel index for torch/torchaudio CUDA installs, e.g. https://mirrors.aliyun.com/pytorch-wheels/cu128/"
          style={{ marginBottom: 0 }}
        >
          <Input
            placeholder="https://mirrors.aliyun.com/pytorch-wheels/cu128/"
            allowClear
          />
        </Form.Item>
      </div>
      {dependencyItems.length > 0 ? (
        dependencyItems.map((item) => {
          const installingThis =
            dependencyInstall?.status === "installing" &&
            dependencyInstall.packages.includes(item.id);
          const dependencyBusy =
            dependencyInstall?.status === "installing" ||
            uninstallingDependency !== null;
          return (
            <div
              key={item.id}
              style={{
                border: "1px solid #f0f0f0",
                borderRadius: 6,
                padding: "12px 12px 8px",
              }}
            >
              <Space direction="vertical" size={4} style={{ width: "100%" }}>
                <Space wrap size={6}>
                  <Typography.Text strong>{item.name}</Typography.Text>
                  <Typography.Text code>{item.spec}</Typography.Text>
                  <Tag color={item.installed ? "success" : "default"}>
                    {item.installed ? "Installed" : "Not installed"}
                  </Tag>
                  {item.required && (
                    <Tag color="gold">Required by current settings</Tag>
                  )}
                </Space>
                <Typography.Text type="secondary">
                  {item.description}
                </Typography.Text>
                <Space>
                  {!item.installed && (
                    <Tooltip
                      title={
                        item.id === "torch_cuda"
                          ? "Replaces the CPU torch wheel with the CUDA 12.8 build from download.pytorch.org."
                          : `Installs ${item.spec} into the QwenPaw interpreter with pip (or uv).`
                      }
                    >
                      <span style={{ display: "inline-flex" }}>
                        <Button
                          size="small"
                          loading={installingThis}
                          disabled={dependencyBusy}
                          onClick={() => void installDependencies([item.id])}
                        >
                          {item.id === "torch_cuda"
                            ? "Install CUDA wheel"
                            : "Install"}
                        </Button>
                      </span>
                    </Tooltip>
                  )}
                  {item.installed && item.can_uninstall && (
                    <Tooltip
                      title={`Runs pip uninstall for ${item.spec}. Imported packages are reloaded so the status refreshes immediately.`}
                    >
                      <span style={{ display: "inline-flex" }}>
                        <Button
                          size="small"
                          danger
                          loading={uninstallingDependency === item.id}
                          disabled={dependencyBusy}
                          onClick={() => void uninstallDependency(item)}
                        >
                          Uninstall
                        </Button>
                      </span>
                    </Tooltip>
                  )}
                </Space>
              </Space>
            </div>
          );
        })
      ) : (
        <Typography.Text type="secondary">
          Loading dependency list…
        </Typography.Text>
      )}
      {requiredMissing.length > 0 && (
        <Alert
          type="warning"
          showIcon
          message={`Missing dependencies required by the current settings: ${requiredMissingItems
            .map((item) => item.name)
            .join(", ")}`}
          description={
            usesQwen3Local
              ? qwen3Device.startsWith("cuda")
                ? "Install qwen-tts, huggingface-hub, and the CUDA PyTorch wheel before using local Qwen3-TTS on a GPU."
                : "Install qwen-tts and huggingface-hub before using the local Qwen3-TTS model."
              : usesLocalModels
              ? "Install qwenpaw[sip] before using offline ASR, wake-word, or Kokoro TTS."
              : "The Local Voice channel and hardware tests need these packages even when using API providers."
          }
          action={
            <Tooltip title="Installs the missing required dependencies with pip (or uv when pip is unavailable).">
              <span style={{ display: "inline-flex" }}>
                <Button
                  size="small"
                  loading={dependencyInstall?.status === "installing"}
                  disabled={dependencyInstall?.status === "installing"}
                  onClick={() => void installDependencies()}
                >
                  Install required dependencies
                </Button>
              </span>
            </Tooltip>
          }
        />
      )}
      {dependencyInstall?.status === "failed" && (
        <Alert
          type="error"
          showIcon
          message={`Dependency installation failed: ${dependencyInstall.packages.join(
            ", ",
          )}`}
          description={dependencyInstall.error}
        />
      )}
      {dependencyInstall?.status === "completed" &&
        dependencyInstall.packages.length > 0 &&
        requiredMissing.length === 0 && (
          <Alert
            type="success"
            showIcon
            message="Local Voice dependencies installed"
          />
        )}
      {usesQwen3Api && (
        <div
          style={{
            border: "1px solid #f0f0f0",
            borderRadius: 8,
            background: "rgba(0, 0, 0, 0.02)",
            padding: "10px 12px",
          }}
        >
          <Typography.Text strong style={{ display: "block" }}>
            Qwen3-TTS voice clone
          </Typography.Text>
          <Typography.Text
            type="secondary"
            style={{ display: "block", margin: "4px 0 8px" }}
          >
            Uses the Clone Reference Audio field above. The returned voice id is
            written into TTS Voice automatically.
          </Typography.Text>
          <Tooltip title="Uploads the reference audio to DashScope and creates a cloned voice on the dedicated Qwen3 VC model.">
            <span style={{ display: "inline-flex" }}>
              <Button
                size="small"
                loading={cloning}
                disabled={cloning}
                onClick={() => void cloneVoice()}
              >
                Clone voice from reference audio
              </Button>
            </span>
          </Tooltip>
        </div>
      )}
      <Typography.Title level={5} style={{ margin: "8px 0 0" }}>
        Hardware tests
      </Typography.Title>
      <Typography.Text type="secondary">
        These tests use the current form values directly and never call an Agent
        or LLM. Keep Local Voice disabled while testing so it does not share the
        microphone or speakers.
      </Typography.Text>
      <Space wrap size={8}>
        {(["wake_word", "asr", "tts"] as LocalVoiceTest[])
          .filter((kind) => kind !== "wake_word" || usesLocalAsr)
          .map((kind) => (
            <Tooltip
              key={kind}
              title={testHint(kind, sttProvider, ttsProvider)}
            >
              <span style={{ display: "inline-flex" }}>
                <Button
                  size="small"
                  type="default"
                  style={{ minWidth: 164 }}
                  loading={test?.status === "running" && test.test === kind}
                  disabled={test?.status === "running"}
                  onClick={() => void startTest(kind)}
                >
                  {testLabel(kind, sttProvider, ttsProvider)}
                </Button>
              </span>
            </Tooltip>
          ))}
      </Space>
      {test && test.status !== "idle" && test.status !== "running" && (
        <Alert
          type={test.status === "passed" ? "success" : "warning"}
          showIcon
          message={
            test.status === "passed"
              ? "Test passed"
              : test.status === "timed_out"
              ? "Test timed out"
              : "Test failed"
          }
          description={
            test.transcript
              ? `Transcript: ${test.transcript}`
              : test.keyword
              ? `Detected wake word: ${test.keyword}`
              : test.message
          }
        />
      )}
      {test?.status === "running" && (
        <Alert type="info" showIcon message="Listening / playing test…" />
      )}
    </Space>
  );
}
