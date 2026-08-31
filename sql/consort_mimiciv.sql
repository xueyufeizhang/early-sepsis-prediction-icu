-- Stage 1: aggregate CONSORT/patient-flow counts for cohort_mimiciv.sql.
-- Parameters and source-project replacement are identical to that query.

WITH ranked_icu AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY subject_id
            ORDER BY intime, stay_id
        ) AS patient_icu_seq
    FROM `{{SOURCE_PROJECT}}.mimiciv_icu.icustays`
),

base AS (
    SELECT
        icu.subject_id,
        icu.stay_id,
        icu.patient_icu_seq,
        (patients.anchor_age
            + DATETIME_DIFF(
                admissions.admittime,
                DATETIME(patients.anchor_year, 1, 1, 0, 0, 0),
                YEAR
            )
        ) AS age,
        DATETIME_DIFF(icu.outtime, icu.intime, SECOND) / 3600.0 AS los_hours,
        DATETIME_DIFF(
            sepsis.suspected_infection_time,
            icu.intime,
            SECOND
        ) / 3600.0 AS onset_offset_h
    FROM ranked_icu AS icu
    INNER JOIN `{{SOURCE_PROJECT}}.mimiciv_hosp.patients` AS patients
        USING (subject_id)
    INNER JOIN `{{SOURCE_PROJECT}}.mimiciv_hosp.admissions` AS admissions
        USING (subject_id, hadm_id)
    LEFT JOIN `{{SOURCE_PROJECT}}.mimiciv_derived.sepsis3` AS sepsis
        USING (subject_id, stay_id)
),

flags AS (
    SELECT
        *,
        age >= @age_min AS is_adult,
        age >= @age_min
            AND (NOT @first_icu_stay_only OR patient_icu_seq = 1) AS is_first_adult,
        age >= @age_min
            AND (NOT @first_icu_stay_only OR patient_icu_seq = 1)
            AND (onset_offset_h IS NULL OR onset_offset_h > @n_hours)
            AS is_prediction_eligible,
        age >= @age_min
            AND (NOT @first_icu_stay_only OR patient_icu_seq = 1)
            AND los_hours >= @n_hours
            AND onset_offset_h > @n_hours
            AND onset_offset_h <= @n_hours + @m_hours
            AS is_positive,
        age >= @age_min
            AND (NOT @first_icu_stay_only OR patient_icu_seq = 1)
            AND los_hours >= @n_hours + @m_hours
            AND (
                onset_offset_h IS NULL
                OR onset_offset_h > @n_hours + @m_hours
            ) AS is_negative
    FROM base
),

stage_masks AS (
    SELECT 1 AS stage_order, 'all_icu_stays' AS stage_code, TRUE AS keep, * FROM flags
    UNION ALL
    SELECT 2, 'adult_stays', is_adult, * FROM flags
    UNION ALL
    SELECT 3, 'first_icu_stays', is_first_adult, * FROM flags
    UNION ALL
    SELECT 4, 'prediction_eligible', is_prediction_eligible, * FROM flags
    UNION ALL
    SELECT 5, 'final_cohort', is_positive OR is_negative, * FROM flags
    UNION ALL
    SELECT 6, 'positive', is_positive, * FROM flags
    UNION ALL
    SELECT 7, 'negative', is_negative, * FROM flags
)

SELECT
    stage_order,
    stage_code,
    CASE stage_code
        WHEN 'all_icu_stays' THEN 'All ICU stays'
        WHEN 'adult_stays' THEN CONCAT('Adults (age >= ', CAST(@age_min AS STRING), ')')
        WHEN 'first_icu_stays' THEN 'First ICU stay per patient'
        WHEN 'prediction_eligible' THEN CONCAT(
            'No Sepsis-3 onset by hour ', CAST(@n_hours AS STRING)
        )
        WHEN 'final_cohort' THEN 'Final labeled cohort'
        WHEN 'positive' THEN CONCAT(
            'Positive: onset in (', CAST(@n_hours AS STRING), ', ',
            CAST(@n_hours + @m_hours AS STRING), '] h'
        )
        WHEN 'negative' THEN CONCAT(
            'Negative: no onset in [0, ',
            CAST(@n_hours + @m_hours AS STRING), '] h'
        )
    END AS stage_label,
    COUNTIF(keep) AS stay_count,
    COUNT(DISTINCT IF(keep, subject_id, NULL)) AS subject_count
FROM stage_masks
GROUP BY stage_order, stage_code
ORDER BY stage_order
