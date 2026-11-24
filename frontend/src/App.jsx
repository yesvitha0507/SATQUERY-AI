import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./App.css";

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const MODES = {
  single: {
    name: "Single Image",
    short: "SINGLE",
    description: "Vision-language analysis of a satellite scene",
    icon: "◉",
  },

  change: {
    name: "Bi-Temporal Change",
    short: "CHANGE",
    description: "Before / after temporal intelligence",
    icon: "◫",
  },

  "optical-sar": {
    name: "Optical + SAR",
    short: "FUSION",
    description: "Cross-modal optical and radar intelligence",
    icon: "◈",
  },
};

function App() {
  // ============================================================
  // STATE
  // ============================================================

  const [mode, setMode] = useState("single");
  const [images, setImages] = useState([]);
  const [question, setQuestion] = useState("");
  const [analysis, setAnalysis] = useState(null);

  const [loading, setLoading] = useState(false);
  const [downloadingReport, setDownloadingReport] = useState(false);

  const [backendOnline, setBackendOnline] = useState(null);

  const [scanProgress, setScanProgress] = useState(0);
  const [activeStage, setActiveStage] = useState(0);

  const [showEvidence, setShowEvidence] = useState(false);
  const [copied, setCopied] = useState(false);

  const currentMode = MODES[mode];

  // ============================================================
  // BACKEND STATUS
  // ============================================================

  useEffect(() => {
    let cancelled = false;

    const checkBackend = async () => {
      try {
        const response = await fetch(`${API_URL}/health`, {
          method: "GET",
        });

        if (!cancelled) {
          setBackendOnline(response.ok);
        }
      } catch {
        if (!cancelled) {
          setBackendOnline(false);
        }
      }
    };

    checkBackend();

    const interval = setInterval(checkBackend, 15000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  // ============================================================
  // ANALYSIS ANIMATION
  // ============================================================

  useEffect(() => {
    if (!loading) {
      setScanProgress(analysis ? 100 : 0);
      return;
    }

    setScanProgress(5);
    setActiveStage(0);

    const progressTimer = setInterval(() => {
      setScanProgress((previous) => {
        if (previous >= 92) return previous;

        return Math.min(
          previous + Math.random() * 8,
          92
        );
      });
    }, 500);

    const stageTimer = setInterval(() => {
      setActiveStage((previous) => {
        return (previous + 1) % 5;
      });
    }, 1200);

    return () => {
      clearInterval(progressTimer);
      clearInterval(stageTimer);
    };
  }, [loading, analysis]);

  // ============================================================
  // FILE VALIDATION
  // ============================================================

  const validateFiles = (files) => {
    for (const file of files) {
      const validType =
        file.type.startsWith("image/") ||
        /\.(tif|tiff)$/i.test(file.name);

      if (!validType) {
        alert(
          `${file.name} is not a supported image.`
        );

        return false;
      }

      if (file.size > 50 * 1024 * 1024) {
        alert(
          `${file.name} is larger than 50 MB.`
        );

        return false;
      }
    }

    return true;
  };

  // ============================================================
  // IMAGE UPLOAD
  // ============================================================

  const handleImageUpload = (
    event,
    imageIndex = null
  ) => {
    const selectedFiles = Array.from(
      event.target.files || []
    );

    if (!selectedFiles.length) return;

    if (!validateFiles(selectedFiles)) {
      event.target.value = "";
      return;
    }

    // ----------------------------------------------------------
    // CHANGE / OPTICAL-SAR INDIVIDUAL UPLOAD
    // ----------------------------------------------------------

    if (
      (mode === "change" ||
        mode === "optical-sar") &&
      imageIndex !== null
    ) {
      const updatedImages = [...images];

      while (updatedImages.length < 2) {
        updatedImages.push(null);
      }

      updatedImages[imageIndex] =
        selectedFiles[0];

      setImages(updatedImages);
      setAnalysis(null);
      setShowEvidence(false);

      event.target.value = "";

      return;
    }

    // ----------------------------------------------------------
    // SINGLE / MULTI IMAGE UPLOAD
    // ----------------------------------------------------------

    const maxImages =
      mode === "single" ? 1 : 2;

    if (selectedFiles.length > maxImages) {
      alert(
        mode === "single"
          ? "Please select only one image."
          : "Please select maximum two images."
      );

      event.target.value = "";

      return;
    }

    setImages(selectedFiles);
    setAnalysis(null);
    setShowEvidence(false);

    event.target.value = "";
  };

  // ============================================================
  // REMOVE IMAGE
  // ============================================================

  const removeImage = (index) => {
    const updated = [...images];

    updated[index] = null;

    if (mode === "single") {
      setImages([]);
    } else {
      setImages(updated);
    }

    setAnalysis(null);
    setShowEvidence(false);
  };

  // ============================================================
  // MODE CHANGE
  // ============================================================

  const changeMode = (newMode) => {
    setMode(newMode);

    setImages([]);
    setAnalysis(null);
    setQuestion("");

    setShowEvidence(false);

    setScanProgress(0);
    setActiveStage(0);
  };

  // ============================================================
  // MODE MODALITY
  // ============================================================

  const getModality = () => {
    if (mode === "optical-sar") {
      return "optical-sar";
    }

    if (mode === "change") {
      return "multitemporal";
    }

    return "auto";
  };

  // ============================================================
  // ANALYZE
  // ============================================================

  const handleAnalyze = async () => {
    // ----------------------------------------------------------
    // IMAGE VALIDATION
    // ----------------------------------------------------------

    if (
      images.length === 0 ||
      images.some((image) => !image)
    ) {
      setAnalysis({
        error:
          mode === "change"
            ? "Please upload both BEFORE and AFTER satellite images."
            : mode === "optical-sar"
            ? "Please upload both OPTICAL and SAR images."
            : "Please upload satellite imagery first.",
      });

      return;
    }

    // ----------------------------------------------------------
    // QUESTION VALIDATION
    // ----------------------------------------------------------

    if (!question.trim()) {
      setAnalysis({
        error:
          "Please enter a question about the imagery.",
      });

      return;
    }

    // ----------------------------------------------------------
    // CHANGE VALIDATION
    // ----------------------------------------------------------

    if (
      mode === "change" &&
      images.length !== 2
    ) {
      setAnalysis({
        error:
          "Bi-temporal analysis requires both BEFORE and AFTER images.",
      });

      return;
    }

    // ----------------------------------------------------------
    // OPTICAL + SAR VALIDATION
    // ----------------------------------------------------------

    if (
      mode === "optical-sar" &&
      images.length !== 2
    ) {
      setAnalysis({
        error:
          "Optical + SAR analysis requires both OPTICAL and SAR images.",
      });

      return;
    }

    // ----------------------------------------------------------
    // FORM DATA
    // ----------------------------------------------------------

    const formData = new FormData();

    images.forEach((image) => {
      if (image) {
        formData.append("files", image);
      }
    });

    formData.append(
      "question",
      question
    );

    formData.append(
      "analysis_mode",
      mode
    );

    formData.append(
      "modality",
      getModality()
    );

    // ----------------------------------------------------------
    // API REQUEST
    // ----------------------------------------------------------

    try {
      setLoading(true);
      setAnalysis(null);
      setShowEvidence(false);

      const response = await fetch(
        `${API_URL}/analyze`,
        {
          method: "POST",
          body: formData,
        }
      );

      let data;

      try {
        data = await response.json();
      } catch {
        throw new Error(
          "Backend returned an invalid response."
        );
      }

      if (!response.ok) {
        throw new Error(
          data.detail ||
            "AI analysis failed."
        );
      }

      setScanProgress(100);
      setActiveStage(4);

      setTimeout(() => {
        setAnalysis(data);
      }, 350);

    } catch (error) {
      console.error(
        "SatQuery AI Error:",
        error
      );

      setAnalysis({
        error:
          error.message ||
          "Unable to connect to SatQuery AI backend.",
      });

    } finally {
      setLoading(false);
    }
  };

  // ============================================================
  // REPORT
  // ============================================================

  const downloadReport = async () => {
    if (
      !analysis ||
      analysis.error
    ) {
      return;
    }

    try {
      setDownloadingReport(true);

      const response = await fetch(
        `${API_URL}/report`,
        {
          method: "POST",

          headers: {
            "Content-Type":
              "application/json",
          },

          body: JSON.stringify(
            analysis
          ),
        }
      );

      if (!response.ok) {
        let message =
          "Unable to generate report.";

        try {
          const errorData =
            await response.json();

          message =
            errorData.detail ||
            message;
        } catch {}

        throw new Error(message);
      }

      const blob =
        await response.blob();

      const url =
        window.URL.createObjectURL(
          blob
        );

      const link =
        document.createElement(
          "a"
        );

      link.href = url;

      link.download =
        "SatQuery_AI_Analysis_Report.pdf";

      document.body.appendChild(
        link
      );

      link.click();

      link.remove();

      window.URL.revokeObjectURL(
        url
      );

    } catch (error) {
      alert(error.message);

    } finally {
      setDownloadingReport(false);
    }
  };

  // ============================================================
  // COPY RESULT
  // ============================================================

  const copyAnswer = async () => {
    if (!analysis?.answer) {
      return;
    }

    try {
      await navigator.clipboard.writeText(
        analysis.answer
      );

      setCopied(true);

      setTimeout(() => {
        setCopied(false);
      }, 1800);

    } catch {
      alert(
        "Unable to copy the analysis."
      );
    }
  };

  // ============================================================
  // SUGGESTION
  // ============================================================

  const setSuggestion = (text) => {
    setQuestion(text);
  };

  // ============================================================
  // KEYBOARD
  // ============================================================

  const handleKeyDown = (event) => {
    if (
      (event.ctrlKey ||
        event.metaKey) &&
      event.key === "Enter"
    ) {
      event.preventDefault();

      handleAnalyze();
    }
  };

  // ============================================================
  // IMAGE PREVIEW
  // ============================================================

  const renderImagePreview = (
    image,
    large = false
  ) => {
    if (!image) {
      return null;
    }

    const previewable =
      image.type?.startsWith(
        "image/"
      ) &&
      !/\.tiff?$/i.test(
        image.name
      );

    if (previewable) {
      return (
        <img
          className={
            large
              ? "preview-large"
              : ""
          }
          src={URL.createObjectURL(
            image
          )}
          alt={image.name}
        />
      );
    }

    return (
      <div className="tiff-preview">
        <span>GEO</span>
        <strong>TIFF</strong>
      </div>
    );
  };

  // ============================================================
  // IMAGE SIZE
  // ============================================================

  const formatFileSize = (
    bytes
  ) => {
    if (!bytes) {
      return "0 KB";
    }

    const mb =
      bytes /
      (1024 * 1024);

    if (mb >= 1) {
      return `${mb.toFixed(
        1
      )} MB`;
    }

    return `${Math.max(
      1,
      Math.round(
        bytes / 1024
      )
    )} KB`;
  };

  // ============================================================
  // CONFIDENCE
  // ============================================================

  const confidenceValue =
    useMemo(() => {
      const value =
        analysis?.confidence?.toLowerCase();

      if (value === "high") {
        return 92;
      }

      if (value === "medium") {
        return 68;
      }

      if (value === "low") {
        return 38;
      }

      return 60;
    }, [analysis]);

  // ============================================================
  // PIPELINE
  // ============================================================

  const pipelineStages = [
    "Upload",
    "Understand",
    "Analyze",
    "Evidence",
    "Insight",
  ];

  // ============================================================
  // SUGGESTIONS
  // ============================================================

  const suggestions = {
    single: [
      [
        "Describe land cover",
        "Describe the land-cover and major objects visible in this image.",
      ],

      [
        "Detect objects",
        "Identify the major objects and structures visible in this image.",
      ],

      [
        "Identify features",
        "What are the dominant geographical features in this image?",
      ],

      [
        "Explain scene",
        "Describe the overall scene and explain what makes this area significant.",
      ],
    ],

    change: [
      [
        "Find changes",
        "What changed between these two dates, and where did the change occur?",
      ],

      [
        "Built-up change",
        "Has the built-up area increased, decreased, or remained unchanged?",
      ],

      [
        "Environmental changes",
        "Describe vegetation and water changes between the two images.",
      ],
    ],

    "optical-sar": [
      [
        "Built-up + water",
        "Use the optical and SAR images together to identify built-up and water-covered regions.",
      ],

      [
        "Compare modalities",
        "What complementary information does the SAR image provide compared with the optical image?",
      ],

      [
        "Fusion insights",
        "Identify areas where optical and SAR observations provide complementary evidence.",
      ],
    ],
  };

  // ============================================================
  // RENDER UPLOAD BOX
  // ============================================================

  const renderUploadBox = ({
    index,
    label,
    title,
    description,
    inputId,
    buttonText,
  }) => {
    const image =
      images[index];

    return (
      <div
        className={`special-upload-box ${
          image
            ? "has-image"
            : ""
        }`}
      >
        <div className="upload-corner top-left" />
        <div className="upload-corner top-right" />
        <div className="upload-corner bottom-left" />
        <div className="upload-corner bottom-right" />

        <div className="special-label">
          <span className="label-dot" />

          {label}
        </div>

        {image ? (
          <div className="uploaded-special">

            <div className="special-preview">
              {renderImagePreview(
                image
              )}

              <div className="scan-line-small" />
            </div>

            <div className="special-file-info">

              <strong>
                {title}
              </strong>

              <span
                title={image.name}
              >
                {image.name}
              </span>

              <small>
                {formatFileSize(
                  image.size
                )}{" "}
                · READY
              </small>

            </div>

            <button
              className="remove-image"
              onClick={() =>
                removeImage(
                  index
                )
              }
              aria-label={`Remove ${label}`}
            >
              ×
            </button>

          </div>
        ) : (
          <div className="empty-special">

            <div className="upload-orbit">

              <div className="upload-orbit-ring" />

              <div className="upload-symbol">
                ↑
              </div>

            </div>

            <h3>
              {title}
            </h3>

            <p>
              {description}
            </p>

            <input
              type="file"
              id={inputId}
              hidden
              accept=".png,.jpg,.jpeg,.tif,.tiff"
              onChange={(event) =>
                handleImageUpload(
                  event,
                  index
                )
              }
            />

            <label
              htmlFor={inputId}
              className="upload-button"
            >
              {buttonText}
            </label>

          </div>
        )}
      </div>
    );
  };

  // ============================================================
  // RENDER AI PROCESSING
  // ============================================================

  const renderProcessing = () => (
    <div className="processing-screen">

      <div className="radar-container">

        <div className="radar-grid" />

        <div className="radar-ring radar-ring-one" />

        <div className="radar-ring radar-ring-two" />

        <div className="radar-ring radar-ring-three" />

        <div className="radar-sweep" />

        <div className="radar-center">
          <span>✦</span>
        </div>

        <div className="radar-ping ping-one" />

        <div className="radar-ping ping-two" />

        <div className="radar-ping ping-three" />

      </div>

      <div className="processing-copy">

        <span className="processing-kicker">
          SATQUERY AI / INTELLIGENCE ENGINE
        </span>

        <h3>
          Analyzing
          <span className="processing-dots">
            ...
          </span>
        </h3>

        <p>
          Processing satellite evidence through the
          remote-sensing analysis pipeline.
        </p>

      </div>

      <div className="processing-progress">

        <div className="progress-meta">

          <span>
            ANALYSIS PROGRESS
          </span>

          <strong>
            {Math.round(
              scanProgress
            )}
            %
          </strong>

        </div>

        <div className="progress-track">

          <span
            style={{
              width: `${scanProgress}%`,
            }}
          />

        </div>

      </div>

      <div className="processing-stages">

        {[
          "Input validation",
          "Domain adaptation",
          "Vision analysis",
          "Evidence generation",
          "Intelligence synthesis",
        ].map(
          (stage, index) => (
            <div
              className={`processing-stage ${
                activeStage >=
                index
                  ? "active"
                  : ""
              }`}
              key={stage}
            >

              <span>
                {activeStage >
                index
                  ? "✓"
                  : index + 1}
              </span>

              {stage}

            </div>
          )
        )}

      </div>

    </div>
  );

  // ============================================================
  // RENDER RESULT
  // ============================================================

  const renderResult = () => {
    if (!analysis) {
      return null;
    }

    // ----------------------------------------------------------
    // ERROR
    // ----------------------------------------------------------

    if (analysis.error) {
      return (
        <div className="error-screen">

          <div className="error-icon">
            !
          </div>

          <span className="processing-kicker">
            ANALYSIS INTERRUPTED
          </span>

          <h3>
            Attention Required
          </h3>

          <p>
            {analysis.error}
          </p>

          <button
            className="secondary-action"
            onClick={() =>
              setAnalysis(null)
            }
          >
            Return to Workspace
          </button>

        </div>
      );
    }

    // ----------------------------------------------------------
    // RESULT
    // ----------------------------------------------------------

    return (
      <div className="result-screen">

        <div className="result-topline">

          <div>
            <span className="success-pulse" />

            ANALYSIS COMPLETE
          </div>

          <span>
            {analysis.task ||
              "REMOTE-SENSING VQA"}
          </span>

        </div>

        <div className="result-heading">

          <div>

            <span className="processing-kicker">
              SATQUERY AI / GENERATED INTELLIGENCE
            </span>

            <h3>
              Scene intelligence
              <span>
                {" "}
                synthesized.
              </span>
            </h3>

          </div>

          <div className="confidence-mini">

            <span>
              CONFIDENCE
            </span>

            <strong>
              {analysis.confidence ||
                "Medium"}
            </strong>

          </div>

        </div>

        {/* RESULT META */}

        <div className="result-meta-grid">

          <div className="result-meta-card">
            <span>MODE</span>

            <strong>
              {currentMode.name}
            </strong>
          </div>

          <div className="result-meta-card">
            <span>AI ENGINE</span>

            <strong>
              {analysis.models?.[0]
                ?.name ||
                "Vision AI"}
            </strong>
          </div>

          <div className="result-meta-card">
            <span>
              ADAPTATION
            </span>

            <strong>
              {analysis.adaptation
                ?.status ||
                "Active"}
            </strong>
          </div>

          <div className="result-meta-card">
            <span>TOOLS</span>

            <strong>
              {analysis.tools
                ?.length ||
                0}{" "}
              EXECUTED
            </strong>
          </div>

        </div>

        {/* ANSWER */}

        <div className="answer-card">

          <div className="answer-card-header">

            <div>

              <span className="answer-number">
                01
              </span>

              <span>
                AI GENERATED INTELLIGENCE
              </span>

            </div>

            <button
              className="copy-button"
              onClick={
                copyAnswer
              }
            >
              {copied
                ? "✓ COPIED"
                : "COPY"}
            </button>

          </div>

          <div className="analysis-answer">
             <ReactMarkdown remarkPlugins={[remarkGfm]}>
                 {analysis.answer || "No analysis available."}
               </ReactMarkdown>
          </div>

        </div>

        {/* TAGS */}

        <div className="insight-tags">

          <span className="insight-tag">
            ✓ IMAGE ANALYZED
          </span>

          <span className="insight-tag">
            ✓ QUERY UNDERSTOOD
          </span>

          <span className="insight-tag">
            ✓ REMOTE-SENSING CONTEXT
          </span>

          {analysis.evidence_image && (
            <span className="insight-tag">
              ✓ VISUAL EVIDENCE
            </span>
          )}

        </div>

        {/* CONFIDENCE */}

        <div className="confidence-card">

          <div className="confidence-header">

            <div>

              <span>
                AI CONFIDENCE ESTIMATE
              </span>

              <small>
                Based on visual and computational evidence
              </small>

            </div>

            <strong>
              {analysis.confidence ||
                "Medium"}
            </strong>

          </div>

          <div className="confidence-track">

            <span
              style={{
                width: `${confidenceValue}%`,
              }}
            />

          </div>

          <div className="confidence-scale">
            <span>LOW</span>
            <span>MEDIUM</span>
            <span>HIGH</span>
          </div>

        </div>

        {/* EVIDENCE */}

        {analysis.evidence_image && (
          <div className="evidence-section">

            <div className="evidence-header">

              <div>

                <span className="processing-kicker">
                  04 / COMPUTATIONAL EVIDENCE
                </span>

                <h3>
                  Visual evidence
                </h3>

              </div>

              <button
                className="evidence-toggle"
                onClick={() =>
                  setShowEvidence(
                    !showEvidence
                  )
                }
              >
                {showEvidence
                  ? "CLOSE VIEW"
                  : "EXPAND VIEW"}
              </button>

            </div>

            <div
              className={`evidence-image ${
                showEvidence
                  ? "expanded"
                  : ""
              }`}
              onClick={() =>
                setShowEvidence(
                  true
                )
              }
            >

              <img
                src={`${API_URL}${analysis.evidence_image}`}
                alt="SatQuery visual evidence"
              />

              <div className="evidence-overlay">

                <span>
                  COMPUTATIONAL EVIDENCE
                </span>

                <strong>
                  {mode === "change"
                    ? "CHANGE MAP"
                    : "MULTIMODAL EVIDENCE"}
                </strong>

              </div>

            </div>

          </div>
        )}

        {/* EXECUTION TRACE */}

        <div className="execution-trace">

          <div className="trace-header">

            <div>

              <span className="processing-kicker">
                05 / AGENT TRACE
              </span>

              <h3>
                Workflow execution
              </h3>

            </div>

            <span className="trace-live">
              ● LIVE TRACE
            </span>

          </div>

          <div className="trace-list">

            {(analysis.tools ||
              [])
              .slice(0, 8)
              .map(
                (
                  tool,
                  index
                ) => (
                  <div
                    className="trace-item"
                    key={`${tool}-${index}`}
                  >

                    <span className="trace-index">
                      {String(
                        index + 1
                      ).padStart(
                        2,
                        "0"
                      )}
                    </span>

                    <div className="trace-line" />

                    <div>

                      <strong>
                        {tool}
                      </strong>

                      <small>
                        specialist workflow executed
                      </small>

                    </div>

                    <span className="trace-check">
                      ✓
                    </span>

                  </div>
                )
              )}

          </div>

        </div>

        {/* REPORT */}

        <button
          className="report-button"
          onClick={
            downloadReport
          }
          disabled={
            downloadingReport
          }
        >

          <span>
            {downloadingReport
              ? "Generating Intelligence Report..."
              : "↓ Download Analysis Report"}
          </span>

          <small>
            PDF
          </small>

        </button>

      </div>
    );
  };

  // ============================================================
  // MAIN RENDER
  // ============================================================

  return (
    <div className="app">

      <div className="ambient-grid" />

      <div className="ambient-glow glow-one" />

      <div className="ambient-glow glow-two" />

      {/* ========================================================
          NAVBAR
      ======================================================== */}

      <nav className="navbar">

        <div className="logo">

          <div className="logo-mark">
            <span />
            <span />
            <span />
          </div>

          <div className="logo-text">
            SATQUERY
            <span> AI</span>
          </div>

        </div>

        <div className="nav-links">

          <a href="#dashboard">
            Dashboard
          </a>

          <a href="#workspace">
            Workspace
          </a>

          <a href="#capabilities">
            Capabilities
          </a>

        </div>

        <div
          className={`system-status ${
            backendOnline === false
              ? "offline"
              : ""
          }`}
        >

          <span className="status-dot" />

          {backendOnline === null
            ? "CONNECTING"
            : backendOnline
            ? "SYSTEM ONLINE"
            : "BACKEND OFFLINE"}

        </div>

      </nav>

      <main id="dashboard">

        {/* ======================================================
            HERO
        ====================================================== */}

        <section className="hero">

          <div className="hero-left">

            <div className="hero-eyebrow">

              <span className="eyebrow-line" />

              AGENTIC REMOTE-SENSING INTELLIGENCE

              <span className="eyebrow-live">
                LIVE
              </span>

            </div>

            <h1>
              Explore Earth.
              <br />
              <span>
                Ask Anything.
              </span>
            </h1>

            <p>
              Transform satellite imagery into
              actionable intelligence through natural
              language. SatQuery AI combines vision
              intelligence, remote sensing and
              agentic workflows in one analytical
              workspace.
            </p>

            <div className="hero-actions">

              <a
                href="#workspace"
                className="hero-primary"
              >
                Open Analysis Workspace
                <span>↓</span>
              </a>

              <div className="hero-secondary">

                <span className="hero-secondary-dot" />

                MULTIMODAL EARTH INTELLIGENCE

              </div>

            </div>

          </div>

          {/* HERO VISUAL */}

          <div className="hero-visual">

            <div className="earth-interface">

              <div className="earth-orbit orbit-a" />

              <div className="earth-orbit orbit-b" />

              <div className="earth-orbit orbit-c" />

              <div className="earth-core">

                <div className="earth-lines">

                  <span />
                  <span />
                  <span />
                  <span />
                  <span />

                </div>

                <div className="earth-crosshair">
                  +
                </div>

              </div>

              <div className="earth-coordinate coordinate-a">
                13.08°N
              </div>

              <div className="earth-coordinate coordinate-b">
                80.27°E
              </div>

              <div className="satellite-particle particle-a" />

              <div className="satellite-particle particle-b" />

              <div className="satellite-particle particle-c" />

            </div>

            <div className="visual-label label-top">
              EARTH OBSERVATION
            </div>

            <div className="visual-label label-bottom">

              <span>◉</span>

              AI / RS ENGINE

            </div>

          </div>

        </section>

        {/* ======================================================
            METRICS
        ====================================================== */}

        <section className="hero-metrics">

          <div className="metric">
            <strong>
              03
            </strong>

            <span>
              Analysis Modes
            </span>
          </div>

          <div className="metric-divider" />

          <div className="metric">
            <strong>
              AI
            </strong>

            <span>
              Vision Engine
            </span>
          </div>

          <div className="metric-divider" />

          <div className="metric">
            <strong>
              RS
            </strong>

            <span>
              Domain Adapted
            </span>
          </div>

          <div className="metric-divider" />

          <div className="metric">
            <strong>
              24/7
            </strong>

            <span>
              Earth Intelligence
            </span>
          </div>

        </section>

        {/* ======================================================
            WORKSPACE
        ====================================================== */}

        <section
          className="workspace-section"
          id="workspace"
        >

          <div className="section-heading">

            <div>

              <span className="section-kicker">
                ANALYSIS WORKSPACE / 01
              </span>

              <h2>
                From satellite pixels
                <span>
                  {" "}
                  to intelligence.
                </span>
              </h2>

            </div>

            <p>
              Select an analysis mode, provide imagery,
              ask a natural-language question and let
              SatQuery AI orchestrate the workflow.
            </p>

          </div>

          {/* ==================================================
              MODE SELECTOR
          ================================================== */}

          <div className="mode-selector">

            {Object.entries(
              MODES
            ).map(
              (
                [key, value],
                index
              ) => (
                <button
                  key={key}
                  className={`mode-button ${
                    mode === key
                      ? "active"
                      : ""
                  }`}
                  onClick={() =>
                    changeMode(
                      key
                    )
                  }
                >

                  <div className="mode-button-number">
                    0{index + 1}
                  </div>

                  <div className="mode-button-icon">
                    {value.icon}
                  </div>

                  <div className="mode-button-content">

                    <strong>
                      {value.name}
                    </strong>

                    <small>
                      {value.description}
                    </small>

                  </div>

                  <div className="mode-arrow">
                    →
                  </div>

                </button>
              )
            )}

          </div>

          {/* ==================================================
              PIPELINE
          ================================================== */}

          <div className="pipeline">

            {pipelineStages.map(
              (
                stage,
                index
              ) => {

                const active =
                  index === 0 ||
                  (loading &&
                    index <=
                      activeStage) ||
                  (!loading &&
                    analysis &&
                    !analysis.error &&
                    index <= 4);

                return (
                  <div
                    className="pipeline-wrapper"
                    key={stage}
                  >

                    <div
                      className={`pipeline-step ${
                        active
                          ? "active"
                          : ""
                      } ${
                        index ===
                          activeStage &&
                        loading
                          ? "processing"
                          : ""
                      }`}
                    >

                      <span className="pipeline-dot">

                        {active &&
                        index <
                          activeStage
                          ? "✓"
                          : index + 1}

                      </span>

                      <span>
                        {stage}
                      </span>

                    </div>

                    {index <
                      pipelineStages.length -
                        1 && (
                      <div
                        className={`pipeline-line ${
                          active &&
                          (index <
                            activeStage ||
                            (!loading &&
                              analysis))
                            ? "active"
                            : ""
                        }`}
                      />
                    )}

                  </div>
                );
              }
            )}

          </div>

          {/* ==================================================
              WORKSPACE GRID
          ================================================== */}

          <div className="workspace-grid">

            {/* =================================================
                INPUT PANEL
            ================================================= */}

            <div className="panel input-panel">

              <div className="panel-header">

                <div>

                  <span className="panel-index">
                    01
                  </span>

                  <div>

                    <span className="panel-kicker">
                      REMOTE-SENSING INPUT
                    </span>

                    <h3>
                      {currentMode.name}
                    </h3>

                  </div>

                </div>

                <span className="panel-status">

                  {
                    images.filter(
                      Boolean
                    ).length
                  }
                  /
                  {mode ===
                  "single"
                    ? 1
                    : 2}{" "}
                  READY

                </span>

              </div>

              <div className="panel-body">

                {/* =================================================
                    SINGLE IMAGE
                ================================================= */}

                {mode === "single" ? (

                  images[0] ? (

                    <div className="single-uploaded">

                      <div className="single-preview">

                        {renderImagePreview(
                          images[0],
                          true
                        )}

                        <div className="preview-scan-line" />

                        <div className="preview-corner pc-one" />

                        <div className="preview-corner pc-two" />

                        <div className="preview-corner pc-three" />

                        <div className="preview-corner pc-four" />

                      </div>

                      <div className="single-file-details">

                        <span className="file-status">
                          ● IMAGE READY
                        </span>

                        <h4>
                          {images[0].name}
                        </h4>

                        <p>

                          {formatFileSize(
                            images[0].size
                          )}

                          {" · "}

                          {images[0].type ||
                            "GeoTIFF / Raster"}

                        </p>

                        <div className="file-tags">

                          <span>
                            REMOTE SENSING
                          </span>

                          <span>
                            VISUAL INPUT
                          </span>

                        </div>

                        <button
                          type="button"
                          className="change-file"
                          onClick={() =>
                            document
                              .getElementById(
                                "single-upload"
                              )
                              ?.click()
                          }
                        >
                          Replace image
                        </button>

                        <input
                          id="single-upload"
                          type="file"
                          hidden
                          accept=".png,.jpg,.jpeg,.tif,.tiff"
                          onChange={
                            handleImageUpload
                          }
                        />

                      </div>

                      <button
                        type="button"
                        className="remove-large"
                        onClick={() =>
                          removeImage(
                            0
                          )
                        }
                        aria-label="Remove image"
                      >
                        ×
                      </button>

                    </div>

                  ) : (

                    <div className="main-dropzone">

                      <div className="drop-grid" />

                      <div className="upload-orbit main">

                        <div className="upload-orbit-ring" />

                        <div className="upload-symbol">
                          ↑
                        </div>

                      </div>

                      <span className="drop-kicker">
                        SATELLITE DATA INPUT
                      </span>

                      <h3>
                        Upload satellite imagery
                      </h3>

                      <p>
                        PNG · JPG · JPEG · TIFF · GeoTIFF
                      </p>

                      {/* IMPORTANT:
                          Real input remains hidden,
                          button opens it directly.
                      */}

                      <input
                        id="main-upload"
                        type="file"
                        hidden
                        accept=".png,.jpg,.jpeg,.tif,.tiff"
                        onChange={
                          handleImageUpload
                        }
                      />

                      <button
                        type="button"
                        className="upload-button primary"
                        onClick={() =>
                          document
                            .getElementById(
                              "main-upload"
                            )
                            ?.click()
                        }
                      >
                        + Upload Image
                      </button>

                      <small>
                        Maximum 50 MB per image
                      </small>

                    </div>

                  )

                ) : (

                  /* =================================================
                     DUAL MODES
                  ================================================= */

                  <div className="dual-upload">

                    {renderUploadBox({
                      index: 0,

                      label:
                        mode ===
                        "change"
                          ? "BEFORE"
                          : "OPTICAL",

                      title:
                        mode ===
                        "change"
                          ? "Before Image"
                          : "Optical Image",

                      description:
                        mode ===
                        "change"
                          ? "Earlier satellite observation"
                          : "RGB / multispectral observation",

                      inputId:
                        mode ===
                        "change"
                          ? "before-upload"
                          : "optical-upload",

                      buttonText:
                        mode ===
                        "change"
                          ? "+ Upload Before"
                          : "+ Upload Optical",
                    })}

                    <div className="dual-connector">

                      <div className="connector-line" />

                      <span>
                        {mode ===
                        "change"
                          ? "VS"
                          : "+"}
                      </span>

                      <div className="connector-line" />

                    </div>

                    {renderUploadBox({
                      index: 1,

                      label:
                        mode ===
                        "change"
                          ? "AFTER"
                          : "SAR",

                      title:
                        mode ===
                        "change"
                          ? "After Image"
                          : "SAR Image",

                      description:
                        mode ===
                        "change"
                          ? "Recent satellite observation"
                          : "Synthetic Aperture Radar",

                      inputId:
                        mode ===
                        "change"
                          ? "after-upload"
                          : "sar-upload",

                      buttonText:
                        mode ===
                        "change"
                          ? "+ Upload After"
                          : "+ Upload SAR",
                    })}

                  </div>

                )}

              </div>

            </div>

            {/* =================================================
                AI PANEL
            ================================================= */}

            <div className="panel intelligence-panel">

              <div className="panel-header">

                <div>

                  <span className="panel-index">
                    02
                  </span>

                  <div>

                    <span className="panel-kicker">
                      AGENTIC INTELLIGENCE
                    </span>

                    <h3>
                      Analysis Engine
                    </h3>

                  </div>

                </div>

                <span className="ai-badge">
                  ✦ VISION AI
                </span>

              </div>

              <div className="panel-body intelligence-body">

                {loading
                  ? renderProcessing()
                  : analysis
                  ? renderResult()
                  : (

                    <div className="idle-intelligence">

                      <div className="idle-orb">

                        <div className="idle-ring ring-one" />

                        <div className="idle-ring ring-two" />

                        <div className="idle-ring ring-three" />

                        <div className="idle-core">
                          ✦
                        </div>

                      </div>

                      <span className="processing-kicker">
                        READY FOR QUERY
                      </span>

                      <h3>
                        Your satellite intelligence
                        starts here.
                      </h3>

                      <p>
                        SatQuery AI will validate the
                        imagery, route your question
                        through the appropriate
                        remote-sensing workflow and
                        synthesize evidence-grounded
                        intelligence.
                      </p>

                      <div className="ready-pills">

                        <span>
                          VISION AI
                        </span>

                        <span>
                          REMOTE SENSING
                        </span>

                        <span>
                          AGENTIC WORKFLOW
                        </span>

                      </div>

                    </div>

                  )}

              </div>

            </div>

          </div>

          {/* ==================================================
              QUERY SECTION
          ================================================== */}

          <div className="query-section">

            <div className="query-header">

              <div>

                <span className="section-kicker">
                  NATURAL LANGUAGE INTERFACE
                </span>

                <h2>
                  Ask{" "}
                  <span>
                    SatQuery AI.
                  </span>
                </h2>

              </div>

              <div className="keyboard-hint">

                <kbd>
                  CTRL
                </kbd>

                <span>
                  +
                </span>

                <kbd>
                  ENTER
                </kbd>

                <small>
                  ANALYZE
                </small>

              </div>

            </div>

            {/* =================================================
                NEW QUERY INTERFACE
                Wide bar + centered button underneath
            ================================================= */}

            <div className="query-interface">

              <div className="query-panel">

                <div className="query-icon">
                  ✦
                </div>

                <input
                  type="text"
                  value={question}
                  onChange={(event) =>
                    setQuestion(
                      event.target.value
                    )
                  }
                  onKeyDown={
                    handleKeyDown
                  }
                  placeholder={
                    mode === "change"
                      ? "What changed between these two observations?"
                      : mode ===
                        "optical-sar"
                      ? "What can optical and SAR reveal together?"
                      : "What would you like to know about this satellite image?"
                  }
                  aria-label="Ask SatQuery AI"
                />

              </div>

              <button
                type="button"
                className="analyze-button"
                onClick={
                  handleAnalyze
                }
                disabled={loading}
              >

                {loading ? (
                  <>
                    <span className="button-loader" />

                    Analyzing
                  </>
                ) : (
                  <>
                    Analyze

                    <span>
                      →
                    </span>
                  </>
                )}

              </button>

            </div>

            {/* =================================================
                SUGGESTIONS
            ================================================= */}

            <div className="suggestions">

              <span className="suggestion-label">
                TRY A QUERY
              </span>

              {suggestions[
                mode
              ].map(
                (
                  [label, text]
                ) => (
                  <button
                    key={label}
                    onClick={() =>
                      setSuggestion(
                        text
                      )
                    }
                  >

                    <span>
                      ⌁
                    </span>

                    {label}

                  </button>
                )
              )}

            </div>

          </div>

        </section>

        {/* ======================================================
            CAPABILITIES
        ====================================================== */}

        <section
          className="capabilities"
          id="capabilities"
        >

          <div className="section-heading capabilities-heading">

            <div>

              <span className="section-kicker">
                SYSTEM CAPABILITIES / 02
              </span>

              <h2>
                Intelligence built for
                <span>
                  {" "}
                  Earth observation.
                </span>
              </h2>

            </div>

          </div>

          <div className="capability-grid">

            <div className="capability-card">

              <span className="capability-number">
                01
              </span>

              <div className="capability-icon">
                ◉
              </div>

              <h3>
                Natural Language
                <br />
                Intelligence
              </h3>

              <p>
                Ask questions about satellite scenes
                using everyday language rather than
                specialist image-processing commands.
              </p>

              <div className="capability-line" />

            </div>

            <div className="capability-card">

              <span className="capability-number">
                02
              </span>

              <div className="capability-icon">
                ◈
              </div>

              <h3>
                Optical + SAR
                <br />
                Fusion
              </h3>

              <p>
                Combine spectral, contextual, radar
                backscatter and structural evidence for
                richer scene understanding.
              </p>

              <div className="capability-line" />

            </div>

            <div className="capability-card">

              <span className="capability-number">
                03
              </span>

              <div className="capability-icon">
                ◫
              </div>

              <h3>
                Bi-Temporal
                <br />
                Change Detection
              </h3>

              <p>
                Compare corresponding regions across
                observations to identify meaningful
                temporal changes.
              </p>

              <div className="capability-line" />

            </div>

            <div className="capability-card">

              <span className="capability-number">
                04
              </span>

              <div className="capability-icon">
                ✦
              </div>

              <h3>
                Agentic Remote-
                <br />
                Sensing Workflow
              </h3>

              <p>
                Automatically route each query through
                domain tools, computational evidence and
                vision intelligence.
              </p>

              <div className="capability-line" />

            </div>

          </div>

        </section>

        {/* ======================================================
            CLOSING CTA
        ====================================================== */}

        <section className="closing-section">

          <div className="closing-grid" />

          <span className="section-kicker">
            SATQUERY AI / EARTH INTELLIGENCE
          </span>

          <h2>
            Ask the Earth
            <br />
            <span>
              anything.
            </span>
          </h2>

          <a href="#workspace">

            Start analyzing

            <span>
              →
            </span>

          </a>

        </section>

      </main>

      {/* ========================================================
          FOOTER
      ======================================================== */}

      <footer>

        <div className="footer-brand">

          <div className="logo-mark small">

            <span />
            <span />
            <span />

          </div>

          SATQUERY AI

        </div>

        <div className="footer-center">
          AGENTIC · REMOTE SENSING · VISION INTELLIGENCE
        </div>

        <div className="footer-right">
          © 2026
        </div>

      </footer>

    </div>
  );
}

export default App;