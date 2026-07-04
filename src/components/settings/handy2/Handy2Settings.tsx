import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useSettings } from "@/hooks/useSettings";
import { SettingsGroup, SettingContainer, ToggleSwitch } from "@/components/ui";
import { commands, type Route } from "@/bindings";

const inputClass =
  "w-full rounded-md border border-mid-gray/30 bg-transparent px-3 py-1.5 text-sm outline-none focus:border-mid-gray";
const buttonClass =
  "shrink-0 rounded-md border border-mid-gray/30 px-3 py-1.5 text-sm hover:bg-mid-gray/10 disabled:opacity-50";

/**
 * Handy 2.0 settings: enable hotword routing + live Obsidian memory, manage the
 * Ollama key (OS credential store via keyring), the memory-file path, and the
 * per-route Ollama models. Calls the `handy2` Tauri commands directly.
 */
export const Handy2Settings: React.FC = () => {
  const { t } = useTranslation();
  const { settings, getSetting, refreshSettings } = useSettings();

  const [keyDraft, setKeyDraft] = useState("");
  const [keySaved, setKeySaved] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [correctionHeard, setCorrectionHeard] = useState("");
  const [correctionWrite, setCorrectionWrite] = useState("");
  const [correctionBusy, setCorrectionBusy] = useState(false);

  useEffect(() => {
    commands.hasOllamaKey().then((r) => {
      if (r.status === "ok") setKeySaved(r.data);
    });
  }, []);

  if (!settings) return null;

  const enabled = getSetting("h2_enabled") ?? false;
  const memoryPath = getSetting("h2_memory_file_path") ?? "";
  const routes: Route[] = getSetting("h2_routes") ?? [];

  const toggleEnabled = async (value: boolean) => {
    await commands.setH2Enabled(value);
    await refreshSettings();
  };

  const saveKey = async () => {
    const r = await commands.setOllamaKey(keyDraft);
    if (r.status === "ok") {
      setKeyDraft("");
      setKeySaved(true);
      setStatus(t("settings.handy2.key.saved"));
    } else {
      setStatus(r.error);
    }
  };

  const testConnection = async () => {
    setBusy(true);
    setStatus(null);
    const r = await commands.testOllamaConnection();
    setStatus(r.status === "ok" ? r.data : r.error);
    setBusy(false);
  };

  const saveMemoryPath = async (value: string) => {
    await commands.setH2MemoryPath(value.trim() === "" ? null : value);
    await refreshSettings();
  };

  const addCorrection = async () => {
    setCorrectionBusy(true);
    setStatus(null);
    const r = await commands.appendCorrection(correctionHeard, correctionWrite);
    if (r.status === "ok") {
      setCorrectionHeard("");
      setCorrectionWrite("");
      setStatus("Correction added. It will apply on the next dictation.");
    } else {
      setStatus(r.error);
    }
    setCorrectionBusy(false);
  };

  const saveRouteModel = async (index: number, model: string) => {
    const next = routes.map((r, i) =>
      i === index ? { ...r, ollama_model: model } : r,
    );
    await commands.setH2Routes(next);
    await refreshSettings();
  };

  return (
    <div className="max-w-3xl w-full mx-auto space-y-6">
      <SettingsGroup title={t("settings.handy2.title")}>
        <ToggleSwitch
          checked={enabled}
          onChange={toggleEnabled}
          label={t("settings.handy2.enablement.label")}
          description={t("settings.handy2.enablement.description")}
        />
      </SettingsGroup>

      <SettingsGroup title={t("settings.handy2.ollama.title")}>
        <SettingContainer
          title={t("settings.handy2.key.label")}
          description={t("settings.handy2.key.description")}
          layout="stacked"
        >
          <div className="flex gap-2 w-full">
            <input
              type="password"
              autoComplete="off"
              className={inputClass}
              placeholder={keySaved ? "•••• (saved)" : "ollama_..."}
              value={keyDraft}
              onChange={(e) => setKeyDraft(e.target.value)}
            />
            <button
              type="button"
              className={buttonClass}
              onClick={saveKey}
              disabled={keyDraft.trim() === ""}
            >
              {t("settings.handy2.key.save")}
            </button>
          </div>
        </SettingContainer>
        <button
          type="button"
          className={buttonClass}
          onClick={testConnection}
          disabled={busy || !keySaved}
        >
          {t("settings.handy2.key.test")}
        </button>
      </SettingsGroup>

      <SettingsGroup
        title={t("settings.handy2.memory.title")}
        description={t("settings.handy2.memory.description")}
      >
        <input
          type="text"
          className={inputClass}
          defaultValue={memoryPath}
          placeholder="C:\\Users\\you\\Vault\\handy-memory.md"
          onBlur={(e) => saveMemoryPath(e.target.value)}
        />

        <SettingContainer
          title="Teach a correction"
          description="Adds a row to the Dictation Corrections table. Applies on the next dictation — no restart needed."
          layout="stacked"
        >
          <div className="flex flex-col gap-2 w-full">
            <div className="flex gap-2 w-full">
              <input
                type="text"
                className={inputClass}
                placeholder="Heard (e.g. Sloan officer)"
                value={correctionHeard}
                onChange={(e) => setCorrectionHeard(e.target.value)}
              />
              <input
                type="text"
                className={inputClass}
                placeholder="Should be (e.g. loan officer)"
                value={correctionWrite}
                onChange={(e) => setCorrectionWrite(e.target.value)}
              />
            </div>
            <button
              type="button"
              className={buttonClass}
              onClick={addCorrection}
              disabled={
                correctionBusy ||
                correctionHeard.trim() === "" ||
                correctionWrite.trim() === ""
              }
            >
              Add correction
            </button>
          </div>
        </SettingContainer>
      </SettingsGroup>

      <SettingsGroup
        title={t("settings.handy2.routes.title")}
        description={t("settings.handy2.routes.description")}
      >
        {routes.map((r, index) => (
          <SettingContainer
            key={r.id}
            title={r.id}
            description={r.trigger ?? t("settings.handy2.routes.default")}
            layout="horizontal"
          >
            <input
              type="text"
              className={inputClass}
              defaultValue={r.ollama_model}
              onBlur={(e) => saveRouteModel(index, e.target.value)}
            />
          </SettingContainer>
        ))}
      </SettingsGroup>

      {status ? <p className="text-sm text-center">{status}</p> : null}
    </div>
  );
};
