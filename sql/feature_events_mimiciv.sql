-- Stage 2: leakage-safe MIMIC-IV feature events from the first N ICU hours.

WITH cohort AS (
    {{COHORT_SQL}}
),

vital_wide AS (
    SELECT
        cohort.subject_id,
        cohort.stay_id,
        cohort.hadm_id,
        cohort.intime,
        vitals.charttime,
        CAST(vitals.heart_rate AS FLOAT64) AS heart_rate,
        CAST(vitals.sbp AS FLOAT64) AS sbp,
        vitals.dbp,
        vitals.mbp,
        vitals.resp_rate,
        CAST(vitals.temperature AS FLOAT64) AS temperature,
        vitals.spo2
    FROM cohort
    INNER JOIN `{{SOURCE_PROJECT}}.mimiciv_3_1_derived.vitalsign` AS vitals
        ON vitals.stay_id = cohort.stay_id
       AND vitals.charttime >= cohort.intime
       AND vitals.charttime <= cohort.feature_window_end
),

vital_events AS (
    SELECT
        subject_id,
        stay_id,
        hadm_id,
        charttime,
        DATETIME_DIFF(charttime, intime, SECOND) / 3600.0 AS offset_hours,
        LEAST(
            @n_hours - 1,
            CAST(
                FLOOR(
                    DATETIME_DIFF(charttime, intime, SECOND) / 3600.0
                ) AS INT64
            )
        ) AS hour_bin,
        feature_name,
        value,
        'vitalsign' AS source_table
    FROM vital_wide
    UNPIVOT (
        value FOR feature_name IN (
            heart_rate,
            sbp,
            dbp,
            mbp,
            resp_rate,
            temperature,
            spo2
        )
    )
),

blood_wide AS (
    SELECT
        cohort.subject_id,
        cohort.stay_id,
        cohort.hadm_id,
        cohort.intime,
        blood.charttime,
        blood.wbc,
        blood.platelet,
        blood.hemoglobin
    FROM cohort
    INNER JOIN `{{SOURCE_PROJECT}}.mimiciv_3_1_derived.complete_blood_count` AS blood
        ON blood.hadm_id = cohort.hadm_id
        AND blood.charttime >= cohort.intime
        AND blood.charttime <= cohort.feature_window_end
),

blood_events AS (
    SELECT
        subject_id,
        stay_id,
        hadm_id,
        charttime,
        DATETIME_DIFF(charttime, intime, SECOND) / 3600.0 AS offset_hours,
        LEAST(
            @n_hours - 1,
            CAST(
                FLOOR(
                    DATETIME_DIFF(charttime, intime, SECOND) / 3600.0
                ) AS INT64
            )
        ) AS hour_bin,
        feature_name,
        value,
        'complete_blood_count' AS source_table
    FROM blood_wide
    UNPIVOT (
        value FOR feature_name IN (
            wbc,
            platelet,
            hemoglobin
        )
    )
),

chemistry_wide AS (
    SELECT
        cohort.subject_id,
        cohort.stay_id,
        cohort.hadm_id,
        cohort.intime,
        chemistry.charttime,
        chemistry.creatinine,
        chemistry.bun,
        chemistry.sodium,
        chemistry.potassium,
        chemistry.chloride,
        chemistry.bicarbonate,
        chemistry.glucose
    FROM cohort
    INNER JOIN `{{SOURCE_PROJECT}}.mimiciv_3_1_derived.chemistry` AS chemistry
        ON chemistry.hadm_id = cohort.hadm_id
        AND chemistry.charttime >= cohort.intime
        AND chemistry.charttime <= cohort.feature_window_end
),

chemistry_events AS (
    SELECT
        subject_id,
        stay_id,
        hadm_id,
        charttime,
        DATETIME_DIFF(charttime, intime, SECOND) / 3600.0 AS offset_hours,
        LEAST(
            @n_hours - 1,
            CAST(
                FLOOR(
                    DATETIME_DIFF(charttime, intime, SECOND) / 3600.0
                ) AS INT64
            )
        ) AS hour_bin,
        feature_name,
        value,
        'chemistry' AS source_table
    FROM chemistry_wide
    UNPIVOT (
        value FOR feature_name IN (
            creatinine,
            bun,
            sodium,
            potassium,
            chloride,
            bicarbonate,
            glucose
        )
    )
),

bg_events AS (
    SELECT
        cohort.subject_id,
        cohort.stay_id,
        cohort.hadm_id,
        bg.charttime,
        DATETIME_DIFF(bg.charttime, cohort.intime, SECOND) / 3600.0 AS offset_hours,
        LEAST(
            @n_hours - 1,
            CAST(
                FLOOR(
                    DATETIME_DIFF(bg.charttime, cohort.intime, SECOND) / 3600.0
                ) AS INT64
            )
        ) AS hour_bin,
        'lactate' AS feature_name,
        bg.lactate AS value,
        'bg' AS source_table
    FROM cohort
    INNER JOIN `{{SOURCE_PROJECT}}.mimiciv_3_1_derived.bg` AS bg
        ON bg.hadm_id = cohort.hadm_id
        AND bg.charttime >= cohort.intime
        AND bg.charttime <= cohort.feature_window_end
        AND bg.lactate IS NOT NULL
),

enzyme_events AS (
    SELECT
        cohort.subject_id,
        cohort.stay_id,
        cohort.hadm_id,
        enzyme.charttime,
        DATETIME_DIFF(enzyme.charttime, cohort.intime, SECOND) / 3600.0 AS offset_hours,
        LEAST(
            @n_hours - 1,
            CAST(
                FLOOR(
                    DATETIME_DIFF(enzyme.charttime, cohort.intime, SECOND) / 3600.0
                ) AS INT64
            )
        ) AS hour_bin,
        'bilirubin_total' AS feature_name,
        enzyme.bilirubin_total AS value,
        'enzyme' AS source_table
    FROM cohort
    INNER JOIN `{{SOURCE_PROJECT}}.mimiciv_3_1_derived.enzyme` AS enzyme
        ON enzyme.hadm_id = cohort.hadm_id
        AND enzyme.charttime >= cohort.intime
        AND enzyme.charttime <= cohort.feature_window_end
        AND enzyme.bilirubin_total IS NOT NULL
),

coagulation_events AS (
    SELECT
        cohort.subject_id,
        cohort.stay_id,
        cohort.hadm_id,
        coagulation.charttime,
        DATETIME_DIFF(coagulation.charttime, cohort.intime, SECOND) / 3600.0 AS offset_hours,
        LEAST(
            @n_hours - 1,
            CAST(
                FLOOR(
                    DATETIME_DIFF(coagulation.charttime, cohort.intime, SECOND) / 3600.0
                ) AS INT64
            )
        ) AS hour_bin,
        'inr' AS feature_name,
        coagulation.inr AS value,
        'coagulation' AS source_table
    FROM cohort
    INNER JOIN `{{SOURCE_PROJECT}}.mimiciv_3_1_derived.coagulation` AS coagulation
        ON coagulation.hadm_id = cohort.hadm_id
        AND coagulation.charttime >= cohort.intime
        AND coagulation.charttime <= cohort.feature_window_end
        AND coagulation.inr IS NOT NULL
),

neuro_events AS (
    SELECT
        cohort.subject_id,
        cohort.stay_id,
        cohort.hadm_id,
        gcs.charttime,
        DATETIME_DIFF(gcs.charttime, cohort.intime, SECOND) / 3600.0 AS offset_hours,
        LEAST(
            @n_hours - 1,
            CAST(
                FLOOR(
                    DATETIME_DIFF(gcs.charttime, cohort.intime, SECOND) / 3600.0
                ) AS INT64
            )
        ) AS hour_bin,
        'gcs' AS feature_name,
        CAST(gcs.gcs AS FLOAT64) AS value,
        'gcs' AS source_table
    FROM cohort
    INNER JOIN `{{SOURCE_PROJECT}}.mimiciv_3_1_derived.gcs` AS gcs
        ON gcs.stay_id = cohort.stay_id
        AND gcs.charttime >= cohort.intime
        AND gcs.charttime <= cohort.feature_window_end
        AND gcs.gcs IS NOT NULL
),

urine_events AS (
    SELECT
        cohort.subject_id,
        cohort.stay_id,
        cohort.hadm_id,
        urine.charttime,
        DATETIME_DIFF(urine.charttime, cohort.intime, SECOND) / 3600.0 AS offset_hours,
        LEAST(
            @n_hours - 1,
            CAST(
                FLOOR(
                    DATETIME_DIFF(urine.charttime, cohort.intime, SECOND) / 3600.0
                ) AS INT64
            )
        ) AS hour_bin,
        'urineoutput' AS feature_name,
        CAST(urine.urineoutput AS FLOAT64) AS value,
        'urine_output' AS source_table
    FROM cohort
    INNER JOIN `{{SOURCE_PROJECT}}.mimiciv_3_1_derived.urine_output` AS urine
        ON urine.stay_id = cohort.stay_id
        AND urine.charttime >= cohort.intime
        AND urine.charttime <= cohort.feature_window_end
        AND urine.urineoutput IS NOT NULL
),

-- The derived table stores piecewise-constant infusion intervals. An interval
-- may begin before ICU intime and continue into the feature window, so filter
-- by interval overlap rather than starttime alone.
vasoactive_wide AS (
    SELECT
        cohort.subject_id,
        cohort.stay_id,
        cohort.hadm_id,
        cohort.intime,
        cohort.feature_window_end,
        vasoactive.starttime,
        vasoactive.endtime,
        CAST(vasoactive.dopamine AS FLOAT64) AS dopamine,
        CAST(vasoactive.epinephrine AS FLOAT64) AS epinephrine,
        CAST(vasoactive.norepinephrine AS FLOAT64) AS norepinephrine,
        CAST(vasoactive.phenylephrine AS FLOAT64) AS phenylephrine,
        CAST(vasoactive.vasopressin AS FLOAT64) AS vasopressin,
        CAST(vasoactive.dobutamine AS FLOAT64) AS dobutamine,
        CAST(vasoactive.milrinone AS FLOAT64) AS milrinone
    FROM cohort
    INNER JOIN `{{SOURCE_PROJECT}}.mimiciv_3_1_derived.vasoactive_agent` AS vasoactive
        ON vasoactive.stay_id = cohort.stay_id
        AND vasoactive.endtime > cohort.intime
        AND vasoactive.starttime < cohort.feature_window_end
),

vasoactive_long AS (
    SELECT
        subject_id,
        stay_id,
        hadm_id,
        intime,
        feature_window_end,
        starttime,
        endtime,
        feature_name,
        agent_rate
    FROM vasoactive_wide
    UNPIVOT (
        agent_rate FOR feature_name IN (
            dopamine,
            epinephrine,
            norepinephrine,
            phenylephrine,
            vasopressin,
            dobutamine,
            milrinone
        )
    )
),

vasoactive_hour_segments AS (
    SELECT
        subject_id,
        stay_id,
        hadm_id,
        intime,
        feature_name,
        agent_rate,
        hour_bin,
        GREATEST(
            starttime,
            intime,
            DATETIME_ADD(intime, INTERVAL hour_bin HOUR)
        ) AS overlap_start,
        LEAST(
            endtime,
            feature_window_end,
            DATETIME_ADD(intime, INTERVAL (hour_bin + 1) HOUR)
        ) AS overlap_end
    FROM vasoactive_long
    CROSS JOIN UNNEST(GENERATE_ARRAY(0, @n_hours - 1)) AS hour_bin
    WHERE starttime < DATETIME_ADD(intime, INTERVAL (hour_bin + 1) HOUR)
      AND endtime > DATETIME_ADD(intime, INTERVAL hour_bin HOUR)
),

vasoactive_events AS (
    SELECT
        subject_id,
        stay_id,
        hadm_id,
        DATETIME_ADD(intime, INTERVAL hour_bin HOUR) AS charttime,
        CAST(hour_bin AS FLOAT64) AS offset_hours,
        hour_bin,
        feature_name,
        -- Average exposure over the complete hour. Off-infusion time within
        -- the hour contributes zero; overlapping segments are additive.
        SUM(
            agent_rate
            * DATETIME_DIFF(overlap_end, overlap_start, SECOND)
            / 3600.0
        ) AS value,
        'vasoactive_agent' AS source_table
    FROM vasoactive_hour_segments
    GROUP BY
        subject_id,
        stay_id,
        hadm_id,
        intime,
        hour_bin,
        feature_name
)

SELECT * FROM vital_events
UNION ALL
SELECT * FROM blood_events
UNION ALL
SELECT * FROM chemistry_events
UNION ALL
SELECT * FROM bg_events
UNION ALL
SELECT * FROM enzyme_events
UNION ALL
SELECT * FROM coagulation_events
UNION ALL
SELECT * FROM neuro_events
UNION ALL
SELECT * FROM urine_events
UNION ALL
SELECT * FROM vasoactive_events
