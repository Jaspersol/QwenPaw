import { useCallback, useState } from "react";

import { localVoiceApi } from "@/api/modules/localVoice";

import { useAppMessage } from "./useAppMessage";

export function useQwen3VoiceClone(
  getConfig: () => Record<string, unknown>,
  onCloned: (voiceId: string) => void,
) {
  const { message } = useAppMessage();
  const [cloning, setCloning] = useState(false);

  const cloneVoice = useCallback(async () => {
    setCloning(true);
    try {
      const result = await localVoiceApi.cloneQwen3Voice(getConfig());
      onCloned(result.voice_id);
      message.success(
        `Voice cloned successfully: ${result.voice_id}. The voice id has been set as the TTS voice.`,
      );
    } catch (error) {
      message.error(
        error instanceof Error ? error.message : "Unable to clone voice",
      );
    } finally {
      setCloning(false);
    }
  }, [getConfig, message, onCloned]);

  return { cloning, cloneVoice };
}
