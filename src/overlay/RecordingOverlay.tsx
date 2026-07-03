import { listen } from "@tauri-apps/api/event";
import React, { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  MicrophoneIcon,
  TranscriptionIcon,
  CancelIcon,
} from "../components/icons";
import "./RecordingOverlay.css";
import { commands } from "@/bindings";
import i18n, { syncLanguageFromSettings } from "@/i18n";
import { getLanguageDirection } from "@/lib/utils/rtl";

type OverlayState = "recording" | "transcribing" | "processing";

const RecordingOverlay: React.FC = () => {
  const { t } = useTranslation();
  const [isVisible, setIsVisible] = useState(false);
  const [state, setState] = useState<OverlayState>("recording");
  const [levels, setLevels] = useState<number[]>(Array(16).fill(0));
  const smoothedLevelsRef = useRef<number[]>(Array(16).fill(0));
  const [partialText, setPartialText] = useState("");
  const direction = getLanguageDirection(i18n.language);

  useEffect(() => {
    const setupEventListeners = async () => {
      // Listen for show-overlay event from Rust
      const unlistenShow = await listen("show-overlay", async (event) => {
        // Sync language from settings each time overlay is shown
        await syncLanguageFromSettings();
        const overlayState = event.payload as OverlayState;
        setState(overlayState);
        setIsVisible(true);
        // Every new recording starts with a clean slate, not leftover
        // text from the previous dictation.
        setPartialText("");
      });

      // Listen for hide-overlay event from Rust
      const unlistenHide = await listen("hide-overlay", () => {
        setIsVisible(false);
        setPartialText("");
      });

      // Listen for live partial-transcript updates while streaming
      // segment-by-segment during recording (see transcription.rs).
      const unlistenPartial = await listen<string>(
        "transcript-partial",
        (event) => {
          setPartialText(event.payload);
        },
      );

      // Listen for mic-level updates
      const unlistenLevel = await listen<number[]>("mic-level", (event) => {
        const newLevels = event.payload as number[];

        // Apply smoothing to reduce jitter
        const smoothed = smoothedLevelsRef.current.map((prev, i) => {
          const target = newLevels[i] || 0;
          return prev * 0.7 + target * 0.3; // Smooth transition
        });

        smoothedLevelsRef.current = smoothed;
        setLevels(smoothed.slice(0, 9));
      });

      // Cleanup function
      return () => {
        unlistenShow();
        unlistenHide();
        unlistenLevel();
        unlistenPartial();
      };
    };

    setupEventListeners();
  }, []);

  // Trailing ~8 words of the live partial transcript — the pill is small,
  // so this cannot show a growing paragraph. Punctuation attached to a
  // word is kept attached; word-boundary edge cases aren't worth handling
  // here. The pill's middle slot only fits ~15-18 characters at this font
  // size, so on top of the word cap we also trim from the front by
  // characters — CSS `text-overflow: ellipsis` truncates from the *end*,
  // which would hide the newest word, so we do the truncation ourselves
  // and let ellipsis only serve as a backstop for the rare very-long-word
  // case.
  const partialWords = partialText.split(/\s+/).filter(Boolean);
  const trailingWords = partialWords.slice(-8).join(" ");
  const MAX_PREVIEW_CHARS = 26;
  const partialPreview =
    trailingWords.length > MAX_PREVIEW_CHARS
      ? `…${trailingWords.slice(-(MAX_PREVIEW_CHARS - 1))}`
      : trailingWords;

  const getIcon = () => {
    if (state === "recording") {
      return <MicrophoneIcon />;
    } else {
      return <TranscriptionIcon />;
    }
  };

  return (
    <div
      dir={direction}
      className={`recording-overlay ${isVisible ? "fade-in" : ""}`}
    >
      <div className="overlay-left">{getIcon()}</div>

      <div className="overlay-middle">
        {state === "recording" &&
          (partialPreview ? (
            <div className="partial-transcript-text">{partialPreview}</div>
          ) : (
            <div className="bars-container">
              {levels.map((v, i) => (
                <div
                  key={i}
                  className="bar"
                  style={{
                    height: `${Math.min(20, 4 + Math.pow(v, 0.7) * 16)}px`, // Cap at 20px max height
                    transition: "height 60ms ease-out, opacity 120ms ease-out",
                    opacity: Math.max(0.2, v * 1.7), // Minimum opacity for visibility
                  }}
                />
              ))}
            </div>
          ))}
        {state === "transcribing" && (
          <div className="transcribing-text">{t("overlay.transcribing")}</div>
        )}
        {state === "processing" && (
          <div className="transcribing-text">{t("overlay.processing")}</div>
        )}
      </div>

      <div className="overlay-right">
        {state === "recording" && (
          <div
            className="cancel-button"
            onClick={() => {
              commands.cancelOperation();
            }}
          >
            <CancelIcon />
          </div>
        )}
      </div>
    </div>
  );
};

export default RecordingOverlay;
