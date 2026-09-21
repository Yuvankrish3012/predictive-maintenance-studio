import React, {
  useEffect,
  useMemo,
  useRef,
  useState
} from 'react';

import { createRoot } from 'react-dom/client';

import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  ReferenceLine
} from 'recharts';

import './styles.css';


/* =========================================================
   CONSTANTS
========================================================= */

const DATASETS = [
  'FD001',
  'FD002',
  'FD003',
  'FD004'
];

const SENSORS = Array.from(
  { length: 21 },
  (_, i) => i + 1
);


/* =========================================================
   FORMATTERS
========================================================= */

const fmt = value =>
  value == null ||
  Number.isNaN(Number(value)) ||
  !Number.isFinite(Number(value))
    ? '—'
    : Number(value).toLocaleString(undefined, {
        maximumFractionDigits: 1
      });

const fmt2 = value =>
  value == null ||
  Number.isNaN(Number(value)) ||
  !Number.isFinite(Number(value))
    ? '—'
    : Number(value).toLocaleString(undefined, {
        maximumFractionDigits: 2
      });


/* =========================================================
   API
========================================================= */

async function api(path, options = {}) {
  const isGet =
    !options.method ||
    options.method.toUpperCase() === 'GET';

  const separator =
    path.includes('?') ? '&' : '?';

  const url = isGet
    ? `/api${path}${separator}_t=${Date.now()}`
    : `/api${path}`;

  const response = await fetch(url, {
    ...options,
    cache: 'no-store',
    headers: {
      ...(options.headers || {}),
      ...(isGet
        ? {
            'Cache-Control':
              'no-cache, no-store, must-revalidate'
          }
        : {})
    }
  });

  let data = null;

  try {
    data = await response.json();
  } catch {
    data = null;
  }

  if (!response.ok) {
    throw new Error(
      typeof data?.detail === 'string'
        ? data.detail
        : typeof data?.message === 'string'
        ? data.message
        : `Request failed (${response.status})`
    );
  }

  return data;
}


/* =========================================================
   EMPTY STATE
========================================================= */

function Empty({ children }) {
  return (
    <div className="empty">
      <span>◎</span>
      <p>{children}</p>
    </div>
  );
}


/* =========================================================
   MODEL SCORE HELPERS
========================================================= */

function isScoreObject(value) {
  return (
    value &&
    typeof value === 'object' &&
    (
      value.mae != null ||
      value.rmse != null ||
      value.nasa_score != null
    )
  );
}

function getModelScores(validation) {
  if (!validation) {
    return {};
  }

  const result = {};

  Object.entries(validation).forEach(
    ([name, value]) => {
      if (isScoreObject(value)) {
        result[name] = value;
      }
    }
  );

  return result;
}

function Scores({ scores }) {
  const normalized =
    getModelScores(scores);

  if (
    !normalized ||
    Object.keys(normalized).length === 0
  ) {
    return (
      <Empty>
        No model evaluation data available.
      </Empty>
    );
  }

  return (
    <div className="scroll">
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>MAE ↓</th>
            <th>RMSE ↓</th>
            <th>NASA score ↓</th>
          </tr>
        </thead>

        <tbody>
          {Object.entries(normalized).map(
            ([name, score]) => (
              <tr key={name}>
                <td>
                  {name === 'xgboost'
                    ? 'XGBoost'
                    : name === 'baseline'
                    ? 'Baseline'
                    : name}
                </td>

                <td>
                  {fmt(score?.mae)}
                </td>

                <td>
                  {fmt(score?.rmse)}
                </td>

                <td>
                  {fmt(score?.nasa_score)}
                </td>
              </tr>
            )
          )}
        </tbody>
      </table>
    </div>
  );
}


/* =========================================================
   FEATURE IMPORTANCE NORMALIZER
========================================================= */

function normalizeFeatureImportance(source) {
  if (!source) {
    return [];
  }

  let values = source;

  /*
   * Backend may return:
   *
   * feature_importance: [
   *   {
   *     feature: "...",
   *     importance: 0.4,
   *     gain: 0.4
   *   }
   * ]
   */

  if (
    !Array.isArray(values) &&
    values?.feature_importance
  ) {
    values = values.feature_importance;
  }

  /*
   * Object mapping:
   *
   * {
   *   sensor_11: 0.42,
   *   sensor_4: 0.21
   * }
   */

  if (
    values &&
    typeof values === 'object' &&
    !Array.isArray(values)
  ) {
    values = Object.entries(values).map(
      ([feature, importance]) => ({
        feature,
        importance
      })
    );
  }

  if (!Array.isArray(values)) {
    return [];
  }

  return values
    .map((item, index) => {
      if (
        item == null ||
        typeof item !== 'object'
      ) {
        return null;
      }

      const feature =
        item.feature ??
        item.name ??
        item.feature_name ??
        item.sensor ??
        item.key ??
        `Feature ${index + 1}`;

      const raw =
        item.gain ??
        item.importance ??
        item.value ??
        item.score ??
        0;

      const gain = Number(raw);

      if (
        !Number.isFinite(gain) ||
        gain < 0
      ) {
        return null;
      }

      return {
        feature: String(feature),
        gain
      };
    })
    .filter(Boolean)
    .sort(
      (a, b) => b.gain - a.gain
    )
    .slice(0, 20);
}


/* =========================================================
   APP
========================================================= */

function App() {

  /* =======================================================
     NASA DATASET STATE
  ======================================================= */

  const [datasets, setDatasets] =
    useState([]);

  const [datasetId, setDatasetId] =
    useState('FD001');

  const [metrics, setMetrics] =
    useState(null);

  const [predictions, setPredictions] =
    useState([]);


  /* =======================================================
     GENERAL UI STATE
  ======================================================= */

  const [offline, setOffline] =
    useState(false);

  const [error, setError] =
    useState('');

  const [message, setMessage] =
    useState('');

  const [pending, setPending] =
    useState(false);

  const [training, setTraining] =
    useState(false);

  /*
   * IMPORTANT:
   *
   * training
   *     = training request/process is active
   *
   * trainingComplete
   *     = backend has written usable artifacts
   *
   * Keeping these separate prevents the UI from
   * remaining indefinitely on "Training..."
   */

  const [trainingComplete, setTrainingComplete] =
    useState(false);

  const [engine, setEngine] =
    useState(1);

  const [sensor, setSensor] =
    useState(2);

  const [history, setHistory] =
    useState([]);

  const [threshold, setThreshold] =
    useState(30);


  /* =======================================================
     REFS
  ======================================================= */

  const trainingPollRef =
    useRef(null);

  const trainingRequestRef =
    useRef(null);


  /* =======================================================
     DATA WORKBENCH MODE
  ======================================================= */

  const [workbenchMode, setWorkbenchMode] =
    useState('nasa');


  /* =======================================================
     CUSTOM DATASET STATE
  ======================================================= */

  const [customFile, setCustomFile] =
    useState(null);

  const [customDataset, setCustomDataset] =
    useState(null);

  const [customValidation, setCustomValidation] =
    useState(null);

  const [customMetrics, setCustomMetrics] =
    useState(null);

  const [customPredictions, setCustomPredictions] =
    useState([]);

  const [customHistory, setCustomHistory] =
    useState([]);

  const [customAsset, setCustomAsset] =
    useState('');

  const [customSensor, setCustomSensor] =
    useState('');

  const [customUploading, setCustomUploading] =
    useState(false);

  const [customTraining, setCustomTraining] =
    useState(false);

  const [customValidating, setCustomValidating] =
    useState(false);

  const [customColumnConfig, setCustomColumnConfig] =
    useState({
      asset: '',
      cycle: '',
      target: ''
    });


  /* =======================================================
     CURRENT NASA DATASET
  ======================================================= */

  const currentDataset =
    useMemo(
      () =>
        datasets.find(
          item =>
            item.dataset === datasetId
        ),
      [datasets, datasetId]
    );


  /* =======================================================
     SELECTED ENGINE
  ======================================================= */

  const selected =
    useMemo(
      () =>
        predictions.find(
          row =>
            Number(row.unit) ===
            Number(engine)
        ),
      [predictions, engine]
    );


  /* =======================================================
     REFRESH DATASET REGISTRY
  ======================================================= */

  async function refreshDatasets() {
    const result =
      await api('/datasets');

    const next =
      Array.isArray(result)
        ? result
        : [];

    setDatasets(next);
    setOffline(false);

    return next;
  }


  /* =======================================================
     LOAD RESULTS
  ======================================================= */

  async function loadResults(
    dataset,
    preserveEngine = true
  ) {
    try {
      const [
        metricResult,
        predictionResult
      ] = await Promise.all([
        api(`/metrics/${dataset}`),
        api(`/predictions/${dataset}`)
      ]);

      setMetrics(
        metricResult || null
      );

      const nextPredictions =
        Array.isArray(
          predictionResult
        )
          ? predictionResult
          : [];

      setPredictions(
        nextPredictions
      );

      /*
       * IMPORTANT:
       *
       * Do not automatically jump back to engine 1
       * after training.
       *
       * Keep the engine the user was inspecting.
       */

      if (
        !preserveEngine &&
        nextPredictions.length
      ) {
        setEngine(
          Number(
            nextPredictions[0].unit
          )
        );
      }

      return {
        metrics: metricResult,
        predictions: nextPredictions
      };

    } catch {
      /*
       * Do not destroy an already visible history chart
       * merely because metrics/predictions aren't ready.
       */

      return null;
    }
  }


  /* =======================================================
     INITIAL BACKEND CONNECTION
  ======================================================= */

  useEffect(() => {
    let alive = true;

    async function initialize() {
      try {
        const result =
          await refreshDatasets();

        if (!alive) {
          return;
        }

        const selectedDataset =
          result.find(
            item =>
              item.dataset ===
              datasetId
          );

        if (
          selectedDataset?.trained
        ) {
          const loaded =
            await loadResults(
              datasetId,
              true
            );

          if (
            alive &&
            loaded
          ) {
            setTrainingComplete(
              true
            );
          }
        }

      } catch {
        if (alive) {
          setOffline(true);
        }
      }
    }

    initialize();

    return () => {
      alive = false;
    };
  }, []);


  /* =======================================================
     DATASET REGISTRY POLLING
  ======================================================= */

  useEffect(() => {
    let alive = true;

    const timer =
      setInterval(async () => {
        try {
          const result =
            await refreshDatasets();

          if (!alive) {
            return;
          }

          const current =
            result.find(
              item =>
                item.dataset ===
                datasetId
            );

          /*
           * If the backend has completed training,
           * hydrate the UI immediately.
           */

          if (
            training &&
            current?.trained
          ) {
            const loaded =
              await loadResults(
                datasetId,
                true
              );

            if (
              alive &&
              loaded
            ) {
              setTrainingComplete(
                true
              );

              setTraining(false);
              setPending(false);

              setMessage(
                `${datasetId} model training completed successfully.`
              );

              stopTrainingPoll();
            }
          }

        } catch {
          if (alive) {
            setOffline(true);
          }
        }
      }, 2000);

    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [datasetId, training]);


  /* =======================================================
     DATASET SWITCH
  ======================================================= */

  useEffect(() => {
    let alive = true;

    setError('');
    setMessage('');

    /*
     * DO NOT clear history here.
     *
     * The sensor-history endpoint below will replace it
     * only after a successful response.
     */

    setMetrics(null);
    setPredictions([]);
    setTrainingComplete(false);

    const selectedDataset =
      datasets.find(
        item =>
          item.dataset ===
          datasetId
      );

    if (
      selectedDataset?.trained
    ) {
      loadResults(
        datasetId,
        true
      ).then(result => {
        if (
          alive &&
          result
        ) {
          setTrainingComplete(
            true
          );
        }
      });
    }

    return () => {
      alive = false;
    };
  }, [datasetId]);


  /* =======================================================
     NASA SENSOR HISTORY
  ======================================================= */

  useEffect(() => {
    let alive = true;

    async function loadSensorHistory() {

      try {
        const result =
          await api(
            `/sensors/${datasetId}/${engine}?sensor=sensor_${sensor}`
          );

        /*
         * Only replace the history after a successful
         * response.
         *
         * This is the important fix for the chart
         * disappearing during training.
         */

        if (
          alive &&
          Array.isArray(
            result?.history
          )
        ) {
          setHistory(
            result.history
          );
        }

      } catch {
        /*
         * NEVER clear history on failure.
         *
         * The last successfully loaded history remains
         * visible.
         */
      }
    }

    loadSensorHistory();

    return () => {
      alive = false;
    };

  }, [
    datasetId,
    engine,
    sensor
  ]);


  /* =======================================================
     STOP TRAINING POLL
  ======================================================= */

  function stopTrainingPoll() {
    if (
      trainingPollRef.current
    ) {
      clearInterval(
        trainingPollRef.current
      );

      trainingPollRef.current =
        null;
    }
  }


  /* =======================================================
     TRAINING RESULT POLL
  ======================================================= */

  function startTrainingPoll(dataset) {
    stopTrainingPoll();

    let attempts = 0;

    trainingPollRef.current =
      setInterval(async () => {

        attempts += 1;

        try {
          const registry =
            await api(
              '/datasets'
            );

          const current =
            Array.isArray(
              registry
            )
              ? registry.find(
                  item =>
                    item.dataset ===
                    dataset
                )
              : null;

          /*
           * Backend now explicitly exposes trained/
           * training_completed/status.
           */

          const completed =
            Boolean(
              current?.trained ||
              current?.training_completed ||
              current?.status ===
                'trained' ||
              current?.status ===
                'completed'
            );

          if (completed) {

            const loaded =
              await loadResults(
                dataset,
                true
              );

            if (loaded) {

              setTrainingComplete(
                true
              );

              setTraining(false);
              setPending(false);

              setMessage(
                `${dataset} model training completed successfully.`
              );

              stopTrainingPoll();

              return;
            }
          }

          /*
           * Approximately 30 minutes.
           */

          if (
            attempts >= 900
          ) {
            stopTrainingPoll();

            setTraining(false);
            setPending(false);

            setError(
              'Training is taking longer than expected. The backend may still be processing the model.'
            );
          }

        } catch {
          /*
           * Ignore individual polling errors.
           * The next cycle retries.
           */
        }

      }, 2000);
  }


  /* =======================================================
     TRAIN NASA DATASET
  ======================================================= */

  async function trainDataset(dataset) {

    setPending(true);
    setTraining(true);
    setTrainingComplete(false);

    setError('');
    setMessage(
      `Training ${dataset} — XGBoost is running and evaluation results are being generated…`
    );

    /*
     * IMPORTANT:
     *
     * Do NOT clear:
     *
     * setHistory([])
     * setEngine(...)
     *
     * The sensor explorer remains visible.
     */

    setMetrics(null);
    setPredictions([]);

    startTrainingPoll(dataset);

    const request =
      api(
        `/train/${dataset}`,
        {
          method: 'POST'
        }
      );

    trainingRequestRef.current =
      request;

    try {

      const result =
        await request;

      /*
       * The backend is synchronous in the current
       * implementation, so this normally contains
       * the final metrics.
       */

      if (
        result &&
        typeof result ===
          'object'
      ) {
        setMetrics(result);
      }

      /*
       * Always fetch the authoritative saved predictions.
       */

      const predictionResult =
        await api(
          `/predictions/${dataset}`
        );

      const nextPredictions =
        Array.isArray(
          predictionResult
        )
          ? predictionResult
          : [];

      setPredictions(
        nextPredictions
      );

      /*
       * Keep the currently selected engine if it exists.
       */

      const currentEngineExists =
        nextPredictions.some(
          row =>
            Number(row.unit) ===
            Number(engine)
        );

      if (
        !currentEngineExists &&
        nextPredictions.length
      ) {
        setEngine(
          Number(
            nextPredictions[0].unit
          )
        );
      }

      await refreshDatasets();

      setTrainingComplete(true);
      setTraining(false);
      setPending(false);

      setMessage(
        `${dataset} model training completed successfully.`
      );

      stopTrainingPoll();

    } catch (err) {

      /*
       * It is possible for the backend to finish writing
       * artifacts even if the HTTP request is interrupted.
       *
       * Check the registry one final time before reporting
       * failure.
       */

      try {

        const registry =
          await api(
            '/datasets'
          );

        const current =
          Array.isArray(
            registry
          )
            ? registry.find(
                item =>
                  item.dataset ===
                  dataset
              )
            : null;

        if (
          current?.trained ||
          current?.training_completed
        ) {

          const loaded =
            await loadResults(
              dataset,
              true
            );

          if (loaded) {

            setTrainingComplete(
              true
            );

            setTraining(false);
            setPending(false);

            setMessage(
              `${dataset} model training completed successfully.`
            );

            stopTrainingPoll();

            return;
          }
        }

      } catch {
        /*
         * Continue to normal error handling.
         */
      }

      setError(
        err.message ||
          'Training failed.'
      );

      setTraining(false);
      setPending(false);

      stopTrainingPoll();
    }
  }


  /* =======================================================
     CLEANUP
  ======================================================= */

  useEffect(() => {
    return () => {
      stopTrainingPoll();
    };
  }, []);


  /* =======================================================
     CUSTOM DATASET HELPERS
  ======================================================= */

  function resetCustomState() {

    setCustomFile(null);
    setCustomDataset(null);
    setCustomValidation(null);
    setCustomMetrics(null);
    setCustomPredictions([]);
    setCustomHistory([]);
    setCustomAsset('');
    setCustomSensor('');

    setCustomColumnConfig({
      asset: '',
      cycle: '',
      target: ''
    });
  }

  function updateCustomColumn(
    key,
    value
  ) {
    setCustomColumnConfig(
      current => ({
        ...current,
        [key]: value
      })
    );

    /*
     * Changing schema means previous validation
     * is no longer authoritative.
     */

    setCustomValidation(null);
    setCustomMetrics(null);
    setCustomPredictions([]);
  }

  function customColumnOptions() {
    return Array.isArray(
      customDataset?.columns
    )
      ? customDataset.columns
      : [];
  }

  function customAssets() {
    const values =
      customDataset?.assets ||
      customDataset?.asset_ids ||
      customValidation?.assets ||
      customValidation?.asset_ids ||
      [];

    return Array.isArray(values)
      ? values
      : [];
  }

  function customSensors() {
    const values =
      customDataset?.sensors ||
      customValidation?.sensors ||
      customDataset?.features ||
      customValidation?.features ||
      [];

    return Array.isArray(values)
      ? values
      : [];
  }

  function customDatasetId() {
    return (
      customDataset?.dataset_id ||
      customDataset?.id ||
      null
    );
  }


  /* =======================================================
     CUSTOM UPLOAD
  ======================================================= */

  async function uploadCustomDataset() {

    if (!customFile) {
      setError(
        'Choose a CSV or compatible tabular dataset first.'
      );
      return;
    }

    setCustomUploading(true);
    setError('');
    setMessage('');

    try {

      const form =
        new FormData();

      form.append(
        'file',
        customFile
      );

      const result =
        await api(
          '/custom/upload',
          {
            method: 'POST',
            body: form
          }
        );

      setCustomDataset(
        result
      );

      setCustomValidation(
        null
      );

      setCustomMetrics(
        null
      );

      setCustomPredictions(
        []
      );

      setCustomHistory(
        []
      );

      const detected =
        result?.detected ||
        {};

      setCustomColumnConfig({
        asset:
          detected.asset ||
          result?.asset_column ||
          '',

        cycle:
          detected.cycle ||
          result?.cycle_column ||
          '',

        target:
          detected.target ||
          result?.target_column ||
          ''
      });

      const assets =
        result?.assets ||
        result?.asset_ids ||
        [];

      const sensors =
        result?.sensors ||
        result?.features ||
        [];

      setCustomAsset(
        assets.length
          ? String(assets[0])
          : ''
      );

      setCustomSensor(
        sensors.length
          ? String(sensors[0])
          : ''
      );

      setMessage(
        `${result?.name || customFile.name} loaded. Review the detected columns, then validate the dataset.`
      );

    } catch (err) {

      setError(
        err.message ||
          'Custom dataset upload failed.'
      );

    } finally {

      setCustomUploading(
        false
      );
    }
  }


  /* =======================================================
     CUSTOM VALIDATION
  ======================================================= */

  async function validateCustomDataset() {

    const id =
      customDatasetId();

    if (!id) {
      setError(
        'Upload a custom dataset before validating it.'
      );
      return;
    }

    setCustomValidating(
      true
    );

    setError('');
    setMessage('');

    try {

      const query =
        new URLSearchParams();

      if (
        customColumnConfig.asset
      ) {
        query.set(
          'asset_column',
          customColumnConfig.asset
        );
      }

      if (
        customColumnConfig.cycle
      ) {
        query.set(
          'cycle_column',
          customColumnConfig.cycle
        );
      }

      if (
        customColumnConfig.target
      ) {
        query.set(
          'target_column',
          customColumnConfig.target
        );
      }

      const suffix =
        query.toString()
          ? `?${query.toString()}`
          : '';

      const result =
        await api(
          `/custom/validate/${encodeURIComponent(id)}${suffix}`
        );

      setCustomValidation(
        result
      );

      if (
        result?.detected
      ) {
        setCustomColumnConfig(
          current => ({
            asset:
              result.detected.asset ||
              current.asset,

            cycle:
              result.detected.cycle ||
              current.cycle,

            target:
              result.detected.target ||
              current.target
          })
        );
      }

      const assets =
        result?.assets ||
        result?.asset_ids ||
        [];

      const sensors =
        result?.sensors ||
        result?.features ||
        [];

      if (
        assets.length
      ) {
        setCustomAsset(
          String(assets[0])
        );
      }

      if (
        sensors.length
      ) {
        setCustomSensor(
          String(sensors[0])
        );
      }

      setMessage(
        result?.valid
          ? 'Custom dataset validation passed. It is ready for RUL training.'
          : 'Custom dataset validation returned issues. Review the validation details below.'
      );

    } catch (err) {

      setError(
        err.message ||
          'Custom dataset validation failed.'
      );

    } finally {

      setCustomValidating(
        false
      );
    }
  }


  /* =======================================================
     CUSTOM TRAINING
  ======================================================= */

  async function trainCustomDataset() {

    const id =
      customDatasetId();

    if (!id) {
      setError(
        'Upload a custom dataset first.'
      );
      return;
    }

    /*
     * Do not allow training until validation
     * explicitly succeeds.
     */

    if (
      customValidation?.valid !==
      true
    ) {
      setError(
        'Validate the custom dataset successfully before training.'
      );
      return;
    }

    setCustomTraining(
      true
    );

    setError('');
    setMessage('');

    try {

      const result =
        await api(
          `/custom/train/${encodeURIComponent(id)}`,
          {
            method: 'POST',

            headers: {
              'Content-Type':
                'application/json'
            },

            body:
              JSON.stringify(
                customColumnConfig
              )
          }
        );

      const nextMetrics =
        result?.metrics ||
        result;

      const nextPredictions =
        result?.predictions ||
        result?.fleet ||
        [];

      setCustomMetrics(
        nextMetrics
      );

      setCustomPredictions(
        Array.isArray(
          nextPredictions
        )
          ? nextPredictions
          : []
      );

      if (
        Array.isArray(
          nextPredictions
        ) &&
        nextPredictions.length
      ) {

        const firstAsset =
          getCustomPredictionValue(
            nextPredictions[0],
            [
              'asset',
              'unit',
              'engine_id',
              'unit_id',
              'asset_id'
            ]
          );

        if (
          firstAsset != null
        ) {
          setCustomAsset(
            String(firstAsset)
          );
        }
      }

      setMessage(
        `${customDataset?.name || 'Custom dataset'} training completed.`
      );

    } catch (err) {

      setError(
        err.message ||
          'Custom model training failed.'
      );

    } finally {

      setCustomTraining(
        false
      );
    }
  }


  /* =======================================================
     CUSTOM SENSOR HISTORY
  ======================================================= */

  useEffect(() => {

    let alive = true;

    async function loadCustomSensorHistory() {

      const id =
        customDatasetId();

      if (
        workbenchMode !==
          'custom' ||
        !id ||
        !customAsset ||
        !customSensor
      ) {
        return;
      }

      try {

        const result =
          await api(
            `/custom/sensors/${encodeURIComponent(id)}/${encodeURIComponent(customAsset)}?sensor=${encodeURIComponent(customSensor)}`
          );

        if (
          alive &&
          Array.isArray(
            result?.history
          )
        ) {
          setCustomHistory(
            result.history
          );
        }

      } catch {
        /*
         * Keep previous custom history visible.
         */
      }
    }

    loadCustomSensorHistory();

    return () => {
      alive = false;
    };

  }, [
    workbenchMode,
    customDataset,
    customAsset,
    customSensor
  ]);


  /* =======================================================
     CUSTOM DOWNLOAD
  ======================================================= */

  function downloadCustomPredictions() {

    const id =
      customDatasetId();

    if (!id) {
      setError(
        'Train a custom dataset before downloading predictions.'
      );
      return;
    }

    window.open(
      `/api/custom/download/${encodeURIComponent(id)}`,
      '_blank'
    );
  }


  /* =======================================================
     CUSTOM PREDICTION VALUE
  ======================================================= */

  function getCustomPredictionValue(
    row,
    keys
  ) {
    for (
      const key of keys
    ) {
      if (
        row?.[key] != null
      ) {
        return row[key];
      }
    }

    return null;
  }


  /* =======================================================
     CUSTOM SELECTED ASSET
  ======================================================= */

  const customSelected =
    useMemo(
      () =>
        customPredictions.find(
          row => {

            const asset =
              getCustomPredictionValue(
                row,
                [
                  'asset',
                  'unit',
                  'engine_id',
                  'unit_id',
                  'asset_id'
                ]
              );

            return (
              String(asset) ===
              String(customAsset)
            );
          }
        ),

      [
        customPredictions,
        customAsset
      ]
    );


  /* =======================================================
     NASA DOWNLOAD
  ======================================================= */

  function downloadPredictions() {
    window.open(
      `/api/download/${datasetId}`,
      '_blank'
    );
  }


  /* =======================================================
     NASA DERIVED DATA
  ======================================================= */

  const fleet =
    Array.isArray(predictions)
      ? predictions
      : [];

  const validation =
    metrics?.validation;

  const testEvaluation =
    metrics?.test_evaluation;

  const modelValidation =
    validation
      ? Object.entries(
          validation
        ).filter(
          ([, value]) =>
            isScoreObject(value)
        )
      : [];

  const selectedModel =
    modelValidation.length
      ? [...modelValidation]
          .sort(
            ([, a], [, b]) =>
              Number(
                a?.rmse ??
                  Infinity
              ) -
              Number(
                b?.rmse ??
                  Infinity
              )
          )[0]?.[0]
      : null;


  /* =======================================================
     FEATURE IMPORTANCE
  ======================================================= */

  const featureImportance =
    normalizeFeatureImportance(
      metrics?.feature_importance ??
      metrics?.xgboost?.feature_importance ??
      metrics?.models?.xgboost?.feature_importance ??
      metrics?.model?.feature_importance ??
      []
    );


  /* =======================================================
     FLEET COUNTS
  ======================================================= */

  const thresholdCount =
    fleet.filter(
      row =>
        row.predicted_rul !=
          null &&
        Number(
          row.predicted_rul
        ) <= threshold
    ).length;

  const normalCount =
    fleet.filter(
      row =>
        row.alert ===
        'NORMAL'
    ).length;

  const warningCount =
    fleet.filter(
      row =>
        row.alert ===
        'WARNING'
    ).length;

  const criticalCount =
    fleet.filter(
      row =>
        row.alert ===
        'CRITICAL'
    ).length;


  /* =======================================================
     FINAL EVALUATION POINTS
  ======================================================= */

  const evaluationPoints =
    predictions
      .map(row => ({
        actual:
          Number(
            row.actual_rul
          ),

        predicted:
          Number(
            row.predicted_rul
          )
      }))
      .filter(
        point =>
          Number.isFinite(
            point.actual
          ) &&
          Number.isFinite(
            point.predicted
          )
      );

  const evaluationValues =
    evaluationPoints.flatMap(
      point => [
        point.actual,
        point.predicted
      ]
    );

  const evaluationMin =
    evaluationValues.length
      ? Math.min(
          ...evaluationValues
        )
      : 0;

  const evaluationMax =
    evaluationValues.length
      ? Math.max(
          ...evaluationValues
        )
      : 1;


  /* =======================================================
     RENDER
  ======================================================= */

  return (
    <div className="layout">

      {/* ===================================================
          SIDEBAR
      =================================================== */}

      <aside>

        <a
          className="brand"
          href="#overview"
        >
          ◈

          <span>
            MAINTENANCE
            <small>
              STUDIO / ML LAB
            </small>
          </span>
        </a>

        <nav>

          <a href="#overview">
            Fleet overview
          </a>

          <a href="#data">
            Data workbench
          </a>

          <a href="#models">
            Model laboratory
          </a>

          <a href="#evaluation">
            Final evaluation
          </a>

        </nav>

        <div className="aside-note">

          <span className="pill">
            RESEARCH EDITION
          </span>

          <p>
            C-MAPSS turbofan degradation.
          </p>

          <p>
            Not for operational aircraft maintenance.
          </p>

        </div>

      </aside>


      {/* ===================================================
          CONTENT
      =================================================== */}

      <div className="content">

        {/* HEADER */}

        <header>

          <span>
            PREDICTIVE INTELLIGENCE / {datasetId}
          </span>

          <span>
            {offline
              ? '○ Backend offline'
              : '● Backend connected'}
          </span>

        </header>


        <main>

          {/* =================================================
              HERO
          ================================================= */}

          <section
            id="overview"
            className="hero"
          >

            <div>

              <p className="eyebrow">
                FROM SENSOR HISTORY TO REMAINING LIFE
              </p>

              <h1>
                See degradation.
                <br />
                <em>
                  Plan ahead.
                </em>
              </h1>

              <p>
                Train, compare, and inspect
                remaining-life models.
              </p>

            </div>

            <div
              className="turbine"
              aria-hidden="true"
            >
              ◎
            </div>

          </section>


          {/* NOTICE */}

          <p className="notice">
            Simulated research dataset ·
            Predictions are cycles, not days ·
            Recorded history, not live telemetry
          </p>


          {/* STATUS */}

          {offline && (
            <p
              role="alert"
              className="error"
            >
              Start the backend on port 8000.
              Connection retries automatically.
            </p>
          )}

          {error && (
            <p
              role="alert"
              className="error"
            >
              {error}
            </p>
          )}

          {message && (
            <p
              role="status"
              aria-live="polite"
              className="success"
            >
              {message}
            </p>
          )}


          {/* =================================================
              DATASET REGISTRY
          ================================================= */}

          <section className="panel">

            <p className="eyebrow">
              DATASET REGISTRY
            </p>

            <h2>
              Select C-MAPSS operating regime
            </h2>

            <div className="controls">

              <label>
                Dataset

                <select
                  value={datasetId}
                  onChange={e =>
                    setDatasetId(
                      e.target.value
                    )
                  }
                  disabled={training}
                >

                  {DATASETS.map(
                    dataset => (
                      <option
                        key={dataset}
                        value={dataset}
                      >
                        {dataset}
                      </option>
                    )
                  )}

                </select>

              </label>

              {currentDataset && (
                <p className="hint">

                  {currentDataset.loaded
                    ? currentDataset.valid
                      ? `${currentDataset.train_engines ?? '—'} training engines · ${currentDataset.test_engines ?? '—'} test engines`
                      : currentDataset.error ||
                        'Dataset validation failed.'
                    : 'Dataset files are missing.'}

                </p>
              )}

            </div>

          </section>


          {/* =================================================
              METRICS
          ================================================= */}

          <section className="metrics">

            {[
              [
                'Training engines',

                metrics?.train_engines ??
                  currentDataset?.train_engines,

                'Run-to-failure histories'
              ],

              [
                'Test engines',

                metrics?.test_engines ??
                  currentDataset?.test_engines,

                'Truncated trajectories'
              ],

              [
                'Validation RMSE',

                selectedModel
                  ? validation?.[
                      selectedModel
                    ]?.rmse
                  : null,

                selectedModel ||
                  'Train to measure'
              ],

              [
                'Threshold alerts',

                metrics
                  ? thresholdCount
                  : null,

                `Heuristic: ≤ ${threshold} cycles`
              ]

            ].map(
              ([label, value, detail]) => (

                <article
                  key={label}
                >

                  <span>
                    {label}
                  </span>

                  <strong>
                    {fmt(value)}
                  </strong>

                  <small>
                    {detail}
                  </small>

                </article>

              )
            )}

          </section>


          {/* =================================================
              01 SENSOR / 02 RUL
          ================================================= */}

          <section className="columns">

            {/* SENSOR EXPLORER */}

            <article className="panel">

              <p className="eyebrow">
                01 / SENSOR EXPLORER
              </p>

              <h2>
                Recorded engine history
              </h2>

              <div className="controls">

                <label>
                  Engine

                  <select
                    disabled={
                      !currentDataset
                    }
                    value={engine}
                    onChange={e =>
                      setEngine(
                        Number(
                          e.target.value
                        )
                      )
                    }
                  >

                    {Array.from(
                      {
                        length:
                          currentDataset?.train_engines ||
                          1
                      },
                      (_, i) =>
                        i + 1
                    ).map(
                      unit => (
                        <option
                          key={unit}
                          value={unit}
                        >
                          Engine {unit}
                        </option>
                      )
                    )}

                  </select>

                </label>

                <label>
                  Sensor

                  <select
                    value={sensor}
                    onChange={e =>
                      setSensor(
                        Number(
                          e.target.value
                        )
                      )
                    }
                  >

                    {SENSORS.map(
                      number => (
                        <option
                          value={number}
                          key={number}
                        >
                          Sensor {number}
                        </option>
                      )
                    )}

                  </select>

                </label>

              </div>


              {history.length ? (

                <>

                  <div
                    className="chart"
                    role="img"
                    aria-label={`Sensor ${sensor} values over engine ${engine} cycles`}
                  >

                    <ResponsiveContainer>

                      <LineChart
                        data={history}
                      >

                        <CartesianGrid
                          stroke="#293a50"
                          strokeDasharray="3 5"
                        />

                        <XAxis
                          dataKey="cycle"
                        />

                        <YAxis
                          domain={[
                            'auto',
                            'auto'
                          ]}
                          width={70}
                        />

                        <Tooltip
                          contentStyle={{
                            background:
                              '#142239'
                          }}
                        />

                        <Line
                          dataKey="value"
                          stroke="#73e3c4"
                          dot={false}
                          strokeWidth={2}
                          isAnimationActive={false}
                        />

                      </LineChart>

                    </ResponsiveContainer>

                  </div>

                  <p className="hint">

                    X: cycle. Y: original sensor
                    measurement. {history.length}{' '}
                    observations; final value{' '}
                    {fmt(
                      history.at(-1)?.value
                    )}.

                  </p>

                </>

              ) : (

                <Empty>
                  Select an engine to display
                  recorded sensor values.
                </Empty>

              )}

            </article>


            {/* RUL */}

            <article className="panel">

              <p className="eyebrow">
                02 / REMAINING USEFUL LIFE
              </p>

              <h2>
                Engine {engine}
              </h2>

              <div className="rul">

                {fmt(
                  selected?.predicted_rul
                )}

                <small>
                  predicted cycles remaining
                </small>

              </div>

              <p className="hint">

                At final observed cycle{' '}
                {selected?.last_cycle ??
                  '—'}.
                Model:{' '}
                {selectedModel ||
                  'not trained'}.

              </p>

              <label>

                Alert threshold:{' '}
                {threshold} cycles

                <input
                  type="range"
                  min="5"
                  max="150"
                  step="5"
                  value={threshold}
                  onChange={e =>
                    setThreshold(
                      Number(
                        e.target.value
                      )
                    )
                  }
                />

              </label>

              <p>

                {selected?.predicted_rul ==
                null

                  ? 'Awaiting training'

                  : Number(
                      selected.predicted_rul
                    ) <= threshold

                  ? 'Review suggested by threshold'

                  : 'Above your alert threshold'}

              </p>

              <p className="hint">
                Illustrative thresholds, not failure
                probabilities or certified maintenance
                advice.
              </p>

              {metrics && (
                <button
                  className="primary"
                  onClick={
                    downloadPredictions
                  }
                >
                  Download fleet predictions ↓
                </button>
              )}

            </article>

          </section>


          {/* =================================================
              03 DATA WORKBENCH
          ================================================= */}

          <section
            id="data"
            className="panel"
          >

            <p className="eyebrow">
              03 / DATA WORKBENCH
            </p>

            <h2>
              Dataset registry & custom data
            </h2>

            <div
              className="workbench-tabs"
              role="tablist"
              aria-label="Dataset source"
            >

              <button
                type="button"
                className={
                  `workbench-tab ${
                    workbenchMode ===
                    'nasa'
                      ? 'active'
                      : ''
                  }`
                }
                onClick={() => {
                  setWorkbenchMode(
                    'nasa'
                  );
                  setError('');
                  setMessage('');
                }}
                role="tab"
                aria-selected={
                  workbenchMode ===
                  'nasa'
                }
              >
                NASA C-MAPSS
              </button>

              <button
                type="button"
                className={
                  `workbench-tab ${
                    workbenchMode ===
                    'custom'
                      ? 'active'
                      : ''
                  }`
                }
                onClick={() => {
                  setWorkbenchMode(
                    'custom'
                  );
                  setError('');
                  setMessage('');
                }}
                role="tab"
                aria-selected={
                  workbenchMode ===
                  'custom'
                }
              >
                CUSTOM DATASET
              </button>

            </div>


            {/* NASA MODE */}

            {workbenchMode ===
            'nasa' ? (

              <>

                <p className="hint">

                  The backend automatically reads
                  train_FD00X.txt, test_FD00X.txt and
                  RUL_FD00X.txt from your CMAPSSData
                  directory.

                </p>

                <div className="summary">

                  {datasets.map(
                    item => (

                      <span
                        key={
                          item.dataset
                        }
                      >

                        <strong>
                          {item.dataset}
                        </strong>{' '}

                        {item.loaded

                          ? item.valid

                            ? item.trained
                              ? '✓ trained'
                              : '✓ ready'

                            : '✕ invalid'

                          : '✕ missing'}

                      </span>

                    )
                  )}

                </div>

                <p className="hint">

                  No upload is required. The application
                  uses the local NASA C-MAPSS files already
                  present in the project.

                </p>

              </>

            ) : (

              /* CUSTOM MODE */

              <>

                <p className="hint">

                  Bring your own compatible
                  predictive-maintenance time-series
                  dataset. Custom mode expects asset/unit,
                  cycle/time and numeric sensor features;
                  RUL can be an explicit target or inferred
                  from run-to-failure histories.

                </p>

                <div className="custom-upload">

                  <div className="controls">

                    <label className="file-input">

                      Dataset file

                      <input
                        type="file"
                        accept=".csv,.txt,.parquet"
                        onChange={e => {

                          const file =
                            e.target.files?.[0] ||
                            null;

                          setCustomFile(
                            file
                          );

                          setCustomDataset(
                            null
                          );

                          setCustomValidation(
                            null
                          );

                          setCustomMetrics(
                            null
                          );

                          setCustomPredictions(
                            []
                          );

                          setCustomHistory(
                            []
                          );

                        }}

                        disabled={
                          customUploading ||
                          customTraining
                        }
                      />

                    </label>

                    <button
                      className="primary"
                      type="button"
                      disabled={
                        !customFile ||
                        customUploading ||
                        offline
                      }
                      onClick={
                        uploadCustomDataset
                      }
                    >

                      {customUploading
                        ? 'Uploading…'
                        : 'Upload dataset'}

                    </button>

                  </div>

                  <p className="hint">

                    {customFile

                      ? `${customFile.name} · ${(customFile.size / 1024 / 1024).toFixed(2)} MB`

                      : 'CSV is recommended. The backend performs schema detection and validation.'}

                  </p>

                </div>


                {customDataset && (

                  <>

                    {/* CUSTOM SCHEMA */}

                    <div className="custom-schema">

                      <div>
                        <span>
                          Rows
                        </span>

                        <strong>
                          {fmt(
                            customDataset.rows ??
                            customDataset.row_count
                          )}
                        </strong>
                      </div>

                      <div>
                        <span>
                          Assets
                        </span>

                        <strong>
                          {fmt(
                            customDataset.asset_count ??
                            customAssets().length
                          )}
                        </strong>
                      </div>

                      <div>
                        <span>
                          Features
                        </span>

                        <strong>
                          {fmt(
                            customDataset.feature_count ??
                            customSensors().length
                          )}
                        </strong>
                      </div>

                      <div>
                        <span>
                          RUL target
                        </span>

                        <strong>
                          {customDataset.target_inferred
                            ? 'Inferred'
                            : customColumnConfig.target ||
                              'Detected'}
                        </strong>
                      </div>

                    </div>


                    {/* CUSTOM COLUMN SELECTION */}

                    <div className="controls custom-columns">

                      <label>

                        Asset / unit column

                        <select
                          value={
                            customColumnConfig.asset
                          }
                          onChange={e =>
                            updateCustomColumn(
                              'asset',
                              e.target.value
                            )
                          }
                        >

                          <option value="">
                            Auto-detect
                          </option>

                          {customColumnOptions().map(
                            column => (
                              <option
                                key={column}
                                value={column}
                              >
                                {column}
                              </option>
                            )
                          )}

                        </select>

                      </label>


                      <label>

                        Cycle / time column

                        <select
                          value={
                            customColumnConfig.cycle
                          }
                          onChange={e =>
                            updateCustomColumn(
                              'cycle',
                              e.target.value
                            )
                          }
                        >

                          <option value="">
                            Auto-detect
                          </option>

                          {customColumnOptions().map(
                            column => (
                              <option
                                key={column}
                                value={column}
                              >
                                {column}
                              </option>
                            )
                          )}

                        </select>

                      </label>


                      <label>

                        RUL / target column

                        <select
                          value={
                            customColumnConfig.target
                          }
                          onChange={e =>
                            updateCustomColumn(
                              'target',
                              e.target.value
                            )
                          }
                        >

                          <option value="">
                            Infer from run-to-failure data
                          </option>

                          {customColumnOptions().map(
                            column => (
                              <option
                                key={column}
                                value={column}
                              >
                                {column}
                              </option>
                            )
                          )}

                        </select>

                      </label>

                    </div>


                    {/* CUSTOM ACTIONS */}

                    <div className="controls">

                      <button
                        className="primary"
                        type="button"
                        disabled={
                          customValidating ||
                          customTraining
                        }
                        onClick={
                          validateCustomDataset
                        }
                      >

                        {customValidating
                          ? 'Validating…'
                          : 'Validate schema'}

                      </button>


                      <button
                        className="primary"
                        type="button"
                        disabled={
                          customTraining ||
                          customValidating ||
                          customValidation?.valid !==
                            true ||
                          offline
                        }
                        onClick={
                          trainCustomDataset
                        }
                      >

                        {customTraining
                          ? 'Training…'
                          : 'Train custom RUL model'}

                      </button>

                    </div>


                    {/* CUSTOM VALIDATION */}

                    {customValidation && (

                      <div className="custom-validation">

                        <p className="job">

                          {customValidation.valid

                            ? '✓ Dataset is compatible with the RUL pipeline.'

                            : '✕ Dataset needs attention before training.'}

                        </p>


                        {customValidation.errors?.length >
                          0 && (

                          <ul>

                            {customValidation.errors.map(
                              (item, index) => (
                                <li
                                  key={index}
                                >
                                  {item}
                                </li>
                              )
                            )}

                          </ul>

                        )}


                        {customValidation.warnings?.length >
                          0 && (

                          <ul>

                            {customValidation.warnings.map(
                              (item, index) => (
                                <li
                                  key={index}
                                >
                                  {item}
                                </li>
                              )
                            )}

                          </ul>

                        )}

                      </div>

                    )}


                    {/* CUSTOM RESULTS */}

                    {customMetrics && (

                      <>

                        <div className="summary">

                          <span>
                            Validation MAE:{' '}
                            <strong>
                              {fmt2(
                                customMetrics
                                  ?.validation
                                  ?.xgboost
                                  ?.mae ??
                                customMetrics
                                  ?.validation
                                  ?.mae
                              )}
                            </strong>
                          </span>

                          <span>
                            Validation RMSE:{' '}
                            <strong>
                              {fmt2(
                                customMetrics
                                  ?.validation
                                  ?.xgboost
                                  ?.rmse ??
                                customMetrics
                                  ?.validation
                                  ?.rmse
                              )}
                            </strong>
                          </span>

                          <span>
                            Test MAE:{' '}
                            <strong>
                              {fmt2(
                                customMetrics
                                  ?.test_evaluation
                                  ?.mae
                              )}
                            </strong>
                          </span>

                          <span>
                            Test RMSE:{' '}
                            <strong>
                              {fmt2(
                                customMetrics
                                  ?.test_evaluation
                                  ?.rmse
                              )}
                            </strong>
                          </span>

                        </div>


                        {/* CUSTOM ASSET/SENSOR CONTROLS */}

                        <div className="controls">

                          <label>

                            Asset

                            <select
                              value={
                                customAsset
                              }
                              onChange={e =>
                                setCustomAsset(
                                  e.target.value
                                )
                              }
                            >

                              {customPredictions.length ===
                                0 &&
                                customAssets().length ===
                                  0 && (
                                  <option value="">
                                    No assets
                                  </option>
                                )}

                              {[
                                ...new Set([
                                  ...customAssets().map(
                                    String
                                  ),

                                  ...customPredictions.map(
                                    row =>
                                      String(
                                        getCustomPredictionValue(
                                          row,
                                          [
                                            'asset',
                                            'unit',
                                            'engine_id',
                                            'unit_id',
                                            'asset_id'
                                          ]
                                        )
                                      )
                                  )
                                ])
                              ]
                                .filter(
                                  value =>
                                    value &&
                                    value !==
                                      'null' &&
                                    value !==
                                      'undefined'
                                )
                                .map(
                                  asset => (
                                    <option
                                      key={
                                        asset
                                      }
                                      value={
                                        asset
                                      }
                                    >
                                      {asset}
                                    </option>
                                  )
                                )}

                            </select>

                          </label>


                          <label>

                            Sensor

                            <select
                              value={
                                customSensor
                              }
                              onChange={e =>
                                setCustomSensor(
                                  e.target.value
                                )
                              }
                            >

                              {customSensors().map(
                                sensorName => (
                                  <option
                                    key={
                                      sensorName
                                    }
                                    value={
                                      sensorName
                                    }
                                  >
                                    {sensorName}
                                  </option>
                                )
                              )}

                            </select>

                          </label>


                          <button
                            className="primary"
                            type="button"
                            onClick={
                              downloadCustomPredictions
                            }
                            disabled={
                              !customPredictions.length
                            }
                          >
                            Download custom predictions ↓
                          </button>

                        </div>


                        {/* CUSTOM HISTORY */}

                        {customHistory.length >
                          0 && (

                          <div
                            className="chart"
                            role="img"
                            aria-label="Custom dataset sensor history"
                          >

                            <ResponsiveContainer>

                              <LineChart
                                data={
                                  customHistory
                                }
                              >

                                <CartesianGrid
                                  stroke="#293a50"
                                  strokeDasharray="3 5"
                                />

                                <XAxis
                                  dataKey="cycle"
                                />

                                <YAxis
                                  width={70}
                                />

                                <Tooltip
                                  contentStyle={{
                                    background:
                                      '#142239'
                                  }}
                                />

                                <Line
                                  dataKey="value"
                                  stroke="#73e3c4"
                                  dot={false}
                                  strokeWidth={2}
                                  isAnimationActive={
                                    false
                                  }
                                />

                              </LineChart>

                            </ResponsiveContainer>

                          </div>

                        )}


                        {/* CUSTOM SELECTED ASSET */}

                        {customSelected && (

                          <div className="summary">

                            <span>
                              Selected asset:{' '}
                              <strong>
                                {customAsset}
                              </strong>
                            </span>

                            <span>
                              Last cycle:{' '}
                              <strong>
                                {fmt(
                                  getCustomPredictionValue(
                                    customSelected,
                                    [
                                      'last_cycle',
                                      'cycle',
                                      'last_time'
                                    ]
                                  )
                                )}
                              </strong>
                            </span>

                            <span>
                              Predicted RUL:{' '}
                              <strong>
                                {fmt(
                                  getCustomPredictionValue(
                                    customSelected,
                                    [
                                      'predicted_rul',
                                      'prediction',
                                      'rul'
                                    ]
                                  )
                                )}
                              </strong>
                            </span>

                            <span>
                              Actual RUL:{' '}
                              <strong>
                                {fmt(
                                  getCustomPredictionValue(
                                    customSelected,
                                    [
                                      'actual_rul',
                                      'actual'
                                    ]
                                  )
                                )}
                              </strong>
                            </span>

                          </div>

                        )}


                        {/* CUSTOM FLEET */}

                        {customPredictions.length >
                          0 && (

                          <div className="scroll fleet">

                            <table>

                              <thead>

                                <tr>
                                  <th>Asset</th>
                                  <th>Observed cycles</th>
                                  <th>Predicted RUL</th>
                                  <th>Actual RUL</th>
                                  <th>Error</th>
                                  <th>Alert</th>
                                </tr>

                              </thead>

                              <tbody>

                                {customPredictions.map(
                                  (
                                    row,
                                    index
                                  ) => {

                                    const asset =
                                      getCustomPredictionValue(
                                        row,
                                        [
                                          'asset',
                                          'unit',
                                          'engine_id',
                                          'unit_id',
                                          'asset_id'
                                        ]
                                      );

                                    const predicted =
                                      getCustomPredictionValue(
                                        row,
                                        [
                                          'predicted_rul',
                                          'prediction',
                                          'rul'
                                        ]
                                      );

                                    const actual =
                                      getCustomPredictionValue(
                                        row,
                                        [
                                          'actual_rul',
                                          'actual'
                                        ]
                                      );

                                    const errorValue =
                                      getCustomPredictionValue(
                                        row,
                                        [
                                          'error',
                                          'residual'
                                        ]
                                      );

                                    return (

                                      <tr
                                        key={`${asset}-${index}`}
                                        className={
                                          String(
                                            asset
                                          ) ===
                                          String(
                                            customAsset
                                          )
                                            ? 'selected-row'
                                            : ''
                                        }
                                      >

                                        <td>

                                          <button
                                            type="button"
                                            onClick={() =>
                                              setCustomAsset(
                                                String(
                                                  asset
                                                )
                                              )
                                            }
                                          >
                                            {String(
                                              asset
                                            )}
                                          </button>

                                        </td>

                                        <td>
                                          {fmt(
                                            getCustomPredictionValue(
                                              row,
                                              [
                                                'last_cycle',
                                                'cycle',
                                                'last_time'
                                              ]
                                            )
                                          )}
                                        </td>

                                        <td>
                                          {fmt(
                                            predicted
                                          )}
                                        </td>

                                        <td>
                                          {fmt(
                                            actual
                                          )}
                                        </td>

                                        <td>

                                          {errorValue ==
                                          null
                                            ? '—'
                                            : `${
                                                Number(
                                                  errorValue
                                                ) > 0
                                                  ? '+'
                                                  : ''
                                              }${fmt(
                                                errorValue
                                              )}`}

                                        </td>

                                        <td>
                                          {row.alert ||
                                            'NORMAL'}
                                        </td>

                                      </tr>

                                    );
                                  }
                                )}

                              </tbody>

                            </table>

                          </div>

                        )}

                      </>

                    )}

                  </>

                )}

              </>

            )}

          </section>


          {/* =================================================
              04 MODEL LABORATORY
          ================================================= */}

          <section
            id="models"
            className="columns"
          >

            {/* MODEL LAB */}

            <article className="panel">

              <p className="eyebrow">
                04 / MODEL LABORATORY
              </p>

              <h2>
                Baseline vs. XGBoost
              </h2>

              <p className="hint">

                Seed 42; 80/20 engine split; causal
                five-cycle rolling features; validation
                is performed on separate engines.

              </p>


              <button
                className="primary"
                disabled={
                  !currentDataset?.valid ||
                  training ||
                  offline
                }
                onClick={() =>
                  trainDataset(
                    datasetId
                  )
                }
              >

                {training
                  ? `Training ${datasetId}…`
                  : trainingComplete
                  ? `Retrain ${datasetId}`
                  : `Train ${datasetId}`}

              </button>


              <p
                role="status"
                aria-live="polite"
                className="job"
              >

                {training

                  ? `Training ${datasetId} — XGBoost is running and evaluation results are being generated…`

                  : trainingComplete

                  ? `${datasetId} training completed successfully. ${selectedModel || 'XGBoost'} results are available.`

                  : metrics

                  ? `Training complete. ${selectedModel || 'Model'} selected using validation RMSE.`

                  : 'Waiting for training. No fabricated scores.'}

              </p>


              {validation && (

                <>

                  <Scores
                    scores={
                      validation
                    }
                  />

                  <p className="hint">

                    Selected:{' '}
                    {selectedModel ||
                      '—'} by validation
                    RMSE, then refitted on all
                    training engines. MAE/RMSE are
                    cycles. NASA score is an
                    asymmetric penalty.

                  </p>

                </>

              )}

            </article>


            {/* FEATURE DIAGNOSTICS */}

            <article className="panel">

              <p className="eyebrow">
                FEATURE DIAGNOSTICS
              </p>

              <h2>
                What XGBoost uses
              </h2>


              {featureImportance.length ? (

                <>

                  <div
                    className="chart importance"
                    role="img"
                    aria-label="Top XGBoost features by normalized split gain"
                  >

                    <ResponsiveContainer>

                      <BarChart
                        data={
                          featureImportance
                        }
                        layout="vertical"
                        margin={{
                          left: 10,
                          right: 20
                        }}
                      >

                        <CartesianGrid
                          stroke="#293a50"
                          strokeDasharray="3 5"
                        />

                        <XAxis
                          type="number"
                        />

                        <YAxis
                          type="category"
                          dataKey="feature"
                          width={140}
                          tick={{
                            fontSize: 10
                          }}
                        />

                        <Tooltip
                          contentStyle={{
                            background:
                              '#142239'
                          }}
                          formatter={value =>
                            [
                              fmt2(
                                value
                              ),
                              'Gain'
                            ]
                          }
                        />

                        <Bar
                          dataKey="gain"
                          fill="#9ca8ff"
                          isAnimationActive={false}
                        />

                      </BarChart>

                    </ResponsiveContainer>

                  </div>

                  <p className="hint">

                    {metrics?.feature_importance_type ||
                      'XGBoost feature importance'}

                  </p>

                </>

              ) : training ? (

                <Empty>
                  Feature importance will appear
                  when XGBoost training finishes.
                </Empty>

              ) : metrics ? (

                <Empty>
                  The model was trained, but the
                  backend did not return feature-importance
                  values.
                </Empty>

              ) : (

                <Empty>
                  Feature diagnostics appear after
                  training.
                </Empty>

              )}

            </article>

          </section>


          {/* =================================================
              05 FINAL TEST EVALUATION
          ================================================= */}

          <section
            id="evaluation"
            className="panel"
          >

            <p className="eyebrow">
              05 / FINAL TEST EVALUATION
            </p>

            <h2>
              Measure generalization
            </h2>

            <p className="hint">

              Final test labels are used only for
              evaluation. They are never used for
              model fitting or model selection.

            </p>


            {testEvaluation ? (

              <div className="columns">

                {/* SCORES */}

                <div>

                  <Scores
                    scores={{
                      [selectedModel ||
                        'Selected model']:
                        testEvaluation
                    }}
                  />

                  <div className="summary">

                    <span>
                      MAE:{' '}
                      <strong>
                        {fmt2(
                          testEvaluation.mae
                        )}
                      </strong>
                    </span>

                    <span>
                      RMSE:{' '}
                      <strong>
                        {fmt2(
                          testEvaluation.rmse
                        )}
                      </strong>
                    </span>

                    <span>
                      NASA:{' '}
                      <strong>
                        {fmt2(
                          testEvaluation.nasa_score
                        )}
                      </strong>
                    </span>

                  </div>

                </div>


                {/* SCATTER */}

                <div>

                  {evaluationPoints.length ? (

                    <>

                      <div
                        className="chart"
                        role="img"
                        aria-label="Actual versus predicted remaining useful life"
                      >

                        <ResponsiveContainer>

                          <ScatterChart>

                            <CartesianGrid
                              stroke="#293a50"
                            />

                            <XAxis
                              type="number"
                              dataKey="actual"
                              name="Actual RUL"
                              domain={[
                                evaluationMin,
                                evaluationMax
                              ]}
                            />

                            <YAxis
                              type="number"
                              dataKey="predicted"
                              name="Predicted RUL"
                              domain={[
                                evaluationMin,
                                evaluationMax
                              ]}
                            />

                            <Tooltip
                              contentStyle={{
                                background:
                                  '#142239'
                              }}
                            />

                            <Scatter
                              data={
                                evaluationPoints
                              }
                              fill="#73e3c4"
                              isAnimationActive={
                                false
                              }
                            />

                            <ReferenceLine
                              segment={[
                                {
                                  x:
                                    evaluationMin,
                                  y:
                                    evaluationMin
                                },
                                {
                                  x:
                                    evaluationMax,
                                  y:
                                    evaluationMax
                                }
                              ]}
                              stroke="#9ca8ff"
                              strokeDasharray="5 5"
                            />

                          </ScatterChart>

                        </ResponsiveContainer>

                      </div>

                      <p className="hint">

                        X: actual remaining cycles.
                        Y: predicted remaining cycles.
                        Dashed line represents perfect
                        prediction.

                      </p>

                    </>

                  ) : (

                    <Empty>
                      Test predictions are available,
                      but there are no valid actual/predicted
                      points to plot.
                    </Empty>

                  )}

                </div>

              </div>

            ) : (

              <Empty>
                Train {datasetId} to generate final
                test evaluation metrics.
              </Empty>

            )}

          </section>


          {/* =================================================
              FLEET REGISTER
          ================================================= */}

          <section className="panel">

            <p className="eyebrow">
              FLEET REGISTER
            </p>

            <h2>
              Engine predictions
            </h2>


            {fleet.length ? (

              <>

                <div className="summary">

                  <span>
                    Normal:{' '}
                    <strong>
                      {normalCount}
                    </strong>
                  </span>

                  <span>
                    Warning:{' '}
                    <strong>
                      {warningCount}
                    </strong>
                  </span>

                  <span>
                    Critical:{' '}
                    <strong>
                      {criticalCount}
                    </strong>
                  </span>

                  <span>
                    Threshold ≤ {threshold}:{' '}
                    <strong>
                      {thresholdCount}
                    </strong>
                  </span>

                </div>


                <div className="scroll fleet">

                  <table>

                    <thead>

                      <tr>
                        <th>Engine</th>
                        <th>Observed cycles</th>
                        <th>Predicted RUL</th>
                        <th>Actual RUL</th>
                        <th>Error</th>
                        <th>Alert</th>
                      </tr>

                    </thead>


                    <tbody>

                      {fleet.map(
                        row => (

                          <tr
                            key={
                              row.unit
                            }
                            className={
                              Number(
                                row.unit
                              ) ===
                              Number(
                                engine
                              )
                                ? 'selected-row'
                                : ''
                            }
                          >

                            <td>

                              <button
                                type="button"
                                onClick={() => {

                                  setEngine(
                                    Number(
                                      row.unit
                                    )
                                  );

                                  document
                                    .getElementById(
                                      'overview'
                                    )
                                    ?.scrollIntoView(
                                      {
                                        behavior:
                                          'smooth'
                                      }
                                    );

                                }}
                              >
                                Engine {row.unit}
                              </button>

                            </td>


                            <td>
                              {fmt(
                                row.last_cycle
                              )}
                            </td>


                            <td>
                              {fmt(
                                row.predicted_rul
                              )}
                            </td>


                            <td>
                              {fmt(
                                row.actual_rul
                              )}
                            </td>


                            <td>

                              {row.error ==
                              null
                                ? '—'
                                : Number(
                                    row.error
                                  ) > 0
                                ? `+${fmt(
                                    row.error
                                  )}`
                                : fmt(
                                    row.error
                                  )}

                            </td>


                            <td>
                              {row.alert ||
                                'NORMAL'}
                            </td>

                          </tr>

                        )
                      )}

                    </tbody>

                  </table>

                </div>

              </>

            ) : (

              <Empty>
                Train {datasetId} to populate the
                fleet register.
              </Empty>

            )}

          </section>

        </main>


        {/* FOOTER */}

        <footer>
          Predictive Maintenance Studio ·
          NASA C-MAPSS · XGBoost · Local research prototype
        </footer>

      </div>

    </div>
  );
}


createRoot(
  document.getElementById('root')
).render(
  <App />
);