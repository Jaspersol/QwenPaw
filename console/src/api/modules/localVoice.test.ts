import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../request", () => ({
  request: vi.fn(),
}));

import { request } from "../request";
import { localVoiceApi } from "./localVoice";

describe("localVoiceApi", () => {
  beforeEach(() => {
    vi.mocked(request).mockReset();
  });

  it("starts the dependency installer for all missing packages", async () => {
    vi.mocked(request).mockResolvedValue({
      status: "accepted",
      message: "started",
    });

    await localVoiceApi.installDependencies();

    expect(request).toHaveBeenCalledWith("/local-voice/dependencies/install", {
      method: "POST",
      body: JSON.stringify({ packages: [], config: {} }),
    });
  });

  it("starts the dependency installer for selected packages", async () => {
    vi.mocked(request).mockResolvedValue({
      status: "accepted",
      message: "started",
    });

    await localVoiceApi.installDependencies(["torch_cuda", "qwen_tts"]);

    expect(request).toHaveBeenCalledWith("/local-voice/dependencies/install", {
      method: "POST",
      body: JSON.stringify({
        packages: ["torch_cuda", "qwen_tts"],
        config: {},
      }),
    });
  });

  it("reads dependency installation progress", async () => {
    const status = {
      status: "installing",
      packages: ["pypinyin"],
      error: null,
    };
    vi.mocked(request).mockResolvedValue(status);

    await expect(localVoiceApi.getDependencyInstallStatus()).resolves.toBe(
      status,
    );
    expect(request).toHaveBeenCalledWith("/local-voice/dependencies/install");
  });
});
