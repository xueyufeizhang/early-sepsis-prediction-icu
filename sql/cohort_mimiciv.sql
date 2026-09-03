-- Stage 1: final MIMIC-IV early-sepsis cohort (BigQuery Standard SQL).
--
-- The Sepsis-3 clinical concept is the validated, materialized MIT-LCP table.
-- This query adds only the project-specific adult, first-stay, and N/M windows.
-- Parameters: @n_hours, @m_hours, @age_min, @first_icu_stay_only.
-- {{SOURCE_PROJECT}} is replaced by src.cohort.render_stage1_sql after validation.

WITH ranked_icu AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY subject_id
            ORDER BY intime, stay_id
        ) AS patient_icu_seq
    FROM `{{SOURCE_PROJECT}}.mimiciv_3_1_icu.icustays`
),

base AS (
    SELECT
        icu.subject_id,
        icu.stay_id,
        icu.hadm_id,
        icu.intime,
        icu.outtime,
        (patients.anchor_age
            + DATETIME_DIFF(
                admissions.admittime,
                DATETIME(patients.anchor_year, 1, 1, 0, 0, 0),
                YEAR
            )
        ) AS age,
        patients.gender,
        admissions.race,
        sepsis.suspected_infection_time AS sepsis_onset_time,
        DATETIME_DIFF(icu.outtime, icu.intime, SECOND) / 3600.0 AS los_hours,
        DATETIME_DIFF(
            sepsis.suspected_infection_time,
            icu.intime,
            SECOND
        ) / 3600.0 AS onset_offset_h
    FROM ranked_icu AS icu
    INNER JOIN `{{SOURCE_PROJECT}}.mimiciv_3_1_hosp.patients` AS patients
        USING (subject_id)
    INNER JOIN `{{SOURCE_PROJECT}}.mimiciv_3_1_hosp.admissions` AS admissions
        USING (subject_id, hadm_id)
    LEFT JOIN `{{SOURCE_PROJECT}}.mimiciv_3_1_derived.sepsis3` AS sepsis
        USING (subject_id, stay_id)
    WHERE (
        NOT @first_icu_stay_only
        OR icu.patient_icu_seq = 1
    )
),

classified AS (
    SELECT
        *,
        CASE
            WHEN los_hours >= @n_hours
                AND onset_offset_h > @n_hours
                AND onset_offset_h <= @n_hours + @m_hours
                THEN 1
            WHEN los_hours >= @n_hours + @m_hours
                AND (
                    onset_offset_h IS NULL
                    OR onset_offset_h > @n_hours + @m_hours
                )
                THEN 0
            ELSE NULL
        END AS label
    FROM base
    WHERE age >= @age_min
)

SELECT
    subject_id,
    stay_id,
    hadm_id,
    intime,
    outtime,
    DATETIME_ADD(intime, INTERVAL @n_hours HOUR) AS feature_window_end,
    DATETIME_ADD(
        intime,
        INTERVAL (@n_hours + @m_hours) HOUR
    ) AS prediction_window_end,
    age,
    gender,
    race,
    los_hours,
    sepsis_onset_time,
    onset_offset_h,
    label
FROM classified
WHERE label IS NOT NULL
ORDER BY subject_id, intime, stay_id