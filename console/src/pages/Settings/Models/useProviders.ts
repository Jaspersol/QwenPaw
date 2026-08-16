import { useState, useEffect, useCallback } from "react";
import api from "../../../api";
import type {
  ProviderInfo,
  ActiveModelsInfo,
  FallbackModelsInfo,
} from "../../../api/types";
import { useAgentStore } from "../../../stores/agentStore";

export function useProviders() {
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [activeModels, setActiveModels] = useState<ActiveModelsInfo | null>(
    null,
  );
  const [fallbackModels, setFallbackModels] =
    useState<FallbackModelsInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { selectedAgent } = useAgentStore();

  const fetchAll = useCallback(async (showLoading = true) => {
    if (showLoading) {
      setLoading(true);
    }
    setError(null);
    try {
      const [provData, activeData, fallbackData] = await Promise.all([
        api.listProviders(),
        api.getActiveModels({ scope: "global" }),
        // Failover chain info may be absent on older backends — never let
        // it break the models page.
        api.getFallbackModels().catch(() => null),
      ]);
      if (!Array.isArray(provData)) {
        throw new Error(
          "Unexpected API response. Is VITE_API_BASE_URL configured correctly?",
        );
      }
      setProviders(provData);
      if (activeData) setActiveModels(activeData);
      if (fallbackData) setFallbackModels(fallbackData);
    } catch (err) {
      const msg =
        err instanceof Error ? err.message : "Failed to load provider data";
      console.error("Failed to load providers:", err);
      setError(msg);
    } finally {
      if (showLoading) {
        setLoading(false);
      }
    }
  }, []);

  // Re-fetch when agent changes to ensure UI stays in sync even though
  // this page uses scope:"global". If future requirements add agent-scoped
  // models, this dependency will be needed.
  useEffect(() => {
    fetchAll();
  }, [fetchAll, selectedAgent]);

  return {
    providers,
    activeModels,
    fallbackModels,
    loading,
    error,
    fetchAll,
  };
}
