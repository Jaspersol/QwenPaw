import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Button, Empty, Popconfirm, Select, Tooltip } from "@agentscope-ai/design";
import {
  ArrowDownOutlined,
  ArrowUpOutlined,
  DeleteOutlined,
  ImportOutlined,
  PlusOutlined,
  SaveOutlined,
} from "@ant-design/icons";
import api from "../../../../../api";
import type {
  FallbackModelEntry,
  ModelSlotConfig,
  ProviderInfo,
} from "../../../../../api/types";
import { useTranslation } from "react-i18next";
import { useAppMessage } from "../../../../../hooks/useAppMessage";
import { getIsConfigured } from "../../utils";
import styles from "../../index.module.less";

interface FallbackModelsSectionProps {
  providers: ProviderInfo[];
  onSaved: () => void;
}

/** Build the display label for a model option. */
function modelOptionLabel(name: string, id: string): string {
  return name && name !== id ? `${name} (${id})` : id;
}

/**
 * Priority-ordered default-model chain ("fallback models") manager.
 *
 * The order of the list IS the failover priority: entry 0 is used first;
 * after three consecutive errors the runtime permanently moves to the next
 * entry and stops when the last entry is also invalid.  An empty list keeps
 * the classic single active-model behaviour.
 */
export const FallbackModelsSection = React.memo(function FallbackModelsSection({
  providers,
  onSaved,
}: FallbackModelsSectionProps) {
  const { t } = useTranslation();
  const { message } = useAppMessage();

  const [models, setModels] = useState<FallbackModelEntry[]>([]);
  const [current, setCurrent] = useState<ModelSlotConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [dirty, setDirty] = useState(false);

  const [addProviderId, setAddProviderId] = useState<string | undefined>();
  const [addModelId, setAddModelId] = useState<string | undefined>();

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.getFallbackModels();
      setModels(data?.models ?? []);
      setCurrent(data?.current ?? null);
    } catch (err) {
      const msg =
        err instanceof Error ? err.message : t("models.fallbackLoadFailed");
      message.error(msg);
    } finally {
      setLoading(false);
    }
  }, [message, t]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Providers that are configured and expose at least one model can be
  // added to the chain manually.
  const eligible = useMemo(
    () =>
      providers.filter((p) => {
        if (!getIsConfigured(p)) return false;
        return (p.models?.length ?? 0) + (p.extra_models?.length ?? 0) > 0;
      }),
    [providers],
  );

  const chosenProvider = eligible.find((p) => p.id === addProviderId);
  const addableModels = useMemo(
    () => [...(chosenProvider?.models ?? []), ...(chosenProvider?.extra_models ?? [])],
    [chosenProvider],
  );

  const existingKeys = useMemo(
    () => new Set(models.map((m) => `${m.provider_id}\u0000${m.model}`)),
    [models],
  );

  const handleProviderChange = (pid: string) => {
    setAddProviderId(pid);
    setAddModelId(undefined);
  };

  const handleAdd = () => {
    if (!addProviderId || !addModelId) return;
    if (existingKeys.has(`${addProviderId}\u0000${addModelId}`)) {
      message.warning(t("models.fallbackAlreadyAdded"));
      return;
    }
    const provider = eligible.find((p) => p.id === addProviderId);
    const model = [...(provider?.models ?? []), ...(provider?.extra_models ?? [])].find(
      (m) => m.id === addModelId,
    );
    setModels((prev) => [
      ...prev,
      {
        provider_id: addProviderId,
        model: addModelId,
        provider_name: provider?.name ?? "",
        model_name: model?.name ?? addModelId,
      },
    ]);
    setAddProviderId(undefined);
    setAddModelId(undefined);
    setDirty(true);
  };

  /** One-click import: append every configured provider's models. */
  const handleImportAll = () => {
    const imported: FallbackModelEntry[] = [];
    for (const provider of providers) {
      if (!getIsConfigured(provider)) continue;
      for (const model of [...(provider.models ?? []), ...(provider.extra_models ?? [])]) {
        const key = `${provider.id}\u0000${model.id}`;
        if (existingKeys.has(key) || imported.some((m) => `${m.provider_id}\u0000${m.model}` === key)) {
          continue;
        }
        imported.push({
          provider_id: provider.id,
          model: model.id,
          provider_name: provider.name,
          model_name: model.name,
        });
      }
    }
    if (imported.length === 0) {
      message.info(t("models.fallbackImportEmpty"));
      return;
    }
    setModels((prev) => [...prev, ...imported]);
    setDirty(true);
    message.success(
      t("models.fallbackImportDone", { count: imported.length }),
    );
  };

  const move = (index: number, delta: -1 | 1) => {
    const target = index + delta;
    if (target < 0 || target >= models.length) return;
    setModels((prev) => {
      const next = [...prev];
      [next[index], next[target]] = [next[target], next[index]];
      return next;
    });
    setDirty(true);
  };

  const remove = (index: number) => {
    setModels((prev) => prev.filter((_, i) => i !== index));
    setDirty(true);
  };

  const canSave = dirty && !loading && !saving;

  const handleSave = async () => {
    setSaving(true);
    try {
      const body = {
        models: models.map((m) => ({
          provider_id: m.provider_id,
          model: m.model,
        })),
      };
      const data = await api.setFallbackModels(body);
      setModels(data?.models ?? []);
      setCurrent(data?.current ?? null);
      setDirty(false);
      message.success(t("models.fallbackSaved"));
      onSaved();
    } catch (err) {
      const msg =
        err instanceof Error ? err.message : t("models.fallbackSaveFailed");
      message.error(msg);
    } finally {
      setSaving(false);
    }
  };

  const currentKey = current
    ? `${current.provider_id}\u0000${current.model}`
    : null;

  return (
    <div className={styles.defaultLlmBody}>
      <p className={styles.llmDescription}>{t("models.fallbackDescription")}</p>

      {/* ---- Add model form ---- */}
      <div className={styles.slotForm}>
        <div className={styles.slotField}>
          <label className={styles.slotLabel}>{t("models.provider")}</label>
          <Select
            style={{ width: "100%" }}
            placeholder={t("models.selectProvider")}
            value={addProviderId}
            onChange={handleProviderChange}
            options={eligible.map((p) => ({
              value: p.id,
              label: p.name,
            }))}
          />
        </div>
        <div className={styles.slotField}>
          <label className={styles.slotLabel}>{t("models.model")}</label>
          <Select
            style={{ width: "100%" }}
            placeholder={t("models.selectModel")}
            disabled={!chosenProvider}
            showSearch
            optionFilterProp="label"
            value={addModelId}
            onChange={(m) => {
              setAddModelId(m);
              setDirty(true);
            }}
            options={addableModels.map((m) => ({
              value: m.id,
              label: modelOptionLabel(m.name, m.id),
            }))}
          />
        </div>
        <div className={[styles.slotField, styles.slotActionField].join(" ")}>
          <label
            className={[styles.slotLabel, styles.visuallyHiddenLabel].join(" ")}
          >
            {t("models.actions")}
          </label>
          <Button
            type="primary"
            disabled={!addProviderId || !addModelId}
            onClick={handleAdd}
            block
            icon={<PlusOutlined />}
          >
            {t("models.fallbackAdd")}
          </Button>
        </div>
      </div>

      {/* ---- One-click import ---- */}
      <div className={styles.fallbackImportRow}>
        <Button
          icon={<ImportOutlined />}
          onClick={handleImportAll}
          disabled={loading}
        >
          {t("models.fallbackImportAll")}
        </Button>
        <span className={styles.fallbackImportHint}>
          {t("models.fallbackImportHint")}
        </span>
      </div>

      {/* ---- Priority list ---- */}
      <div className={styles.fallbackList}>
        {loading ? (
          <Empty description={t("models.loading")} />
        ) : models.length === 0 ? (
          <Empty description={t("models.fallbackEmpty")} />
        ) : (
          models.map((entry, index) => {
            const isCurrent = currentKey === `${entry.provider_id}\u0000${entry.model}`;
            return (
              <div
                key={`${entry.provider_id}\u0000${entry.model}`}
                className={styles.fallbackItem}
              >
                <span className={styles.fallbackIndex}>{index + 1}</span>
                <div className={styles.fallbackItemInfo}>
                  <span className={styles.fallbackItemName}>
                    {entry.provider_name || entry.provider_id}
                    <span className={styles.fallbackItemSep}>/</span>
                    {modelOptionLabel(entry.model_name || "", entry.model)}
                    {isCurrent && (
                      <span className={styles.fallbackCurrentTag}>
                        {t("models.fallbackInUse")}
                      </span>
                    )}
                  </span>
                  <span className={styles.fallbackItemId}>
                    {entry.provider_id} / {entry.model}
                  </span>
                </div>
                <div className={styles.modelListItemActions}>
                  <Tooltip title={t("models.fallbackMoveUp")}>
                    <Button
                      size="small"
                      type="text"
                      icon={<ArrowUpOutlined />}
                      disabled={index === 0}
                      onClick={() => move(index, -1)}
                    />
                  </Tooltip>
                  <Tooltip title={t("models.fallbackMoveDown")}>
                    <Button
                      size="small"
                      type="text"
                      icon={<ArrowDownOutlined />}
                      disabled={index === models.length - 1}
                      onClick={() => move(index, 1)}
                    />
                  </Tooltip>
                  <Popconfirm
                    title={t("models.fallbackRemoveConfirm")}
                    onConfirm={() => remove(index)}
                    okText={t("common.delete")}
                    cancelText={t("common.cancel")}
                  >
                    <Button
                      size="small"
                      type="text"
                      danger
                      icon={<DeleteOutlined />}
                    />
                  </Popconfirm>
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* ---- Actions ---- */}
      <div className={styles.slotActions}>
        <Button
          type="primary"
          loading={saving}
          disabled={!canSave}
          onClick={handleSave}
          icon={<SaveOutlined />}
        >
          {t("models.save")}
        </Button>
      </div>
    </div>
  );
});
