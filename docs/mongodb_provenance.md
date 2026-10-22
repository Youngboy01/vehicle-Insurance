# MongoDB source provenance

The training pipeline treats MongoDB extraction as a controlled, immutable
source boundary:

```text
MongoDB → snapshot-session extraction → feature_store/data.csv
                                     → feature_store/source_manifest.json
                                     → training and release lineage
```

Training and evaluation consume the verified CSV snapshot produced by
ingestion. They do not query MongoDB directly. The CSV and manifest are an
inseparable pair. If one is missing, or if their hashes do not agree, the run
stops instead of silently creating replacement evidence.

## Manifest contents

`source_manifest.json` records:

- `snapshot_id`, manifest/snapshot version, immutable flag, and snapshot file name;
- MongoDB system, database, collection, and namespace;
- the exact query filter and canonical filter representation;
- the explicit schema projection, including `_id: 0`;
- ascending `id` ordering and the supporting index name;
- the snapshot-session read concern, cluster time or operation time, and local
  read/extraction timestamps;
- row count, `id` minimum/maximum, canonical CSV SHA-256, and file SHA-256;
- the retry policy: abort a failed read, and create a new ID only for a new
  extraction request.

The manifest is copied into `DataIngestionArtifact`, nested in trainer lineage,
written into `metrics.json`, included in the evaluation report, and logged to
MLflow as `lineage/source_manifest.json` and as part of
`lineage/lineage.json`. Future release metadata should copy this same lineage
object; it must not invent a new source identity.

## Required MongoDB index

The extraction refuses to use an unindexed sort. Create the supporting index
once in the configured `INSURANCE` database before running training:

```javascript
use INSURANCE
db.INSURANCE_DATA.createIndex(
  { id: 1 },
  { name: "vehicle_insurance_id_ascending" }
)
```

The pipeline uses the discovered index as a query hint. It does not create or
modify database indexes implicitly during training.

### No-code check in MongoDB Atlas

You do not need to use JavaScript or `mongosh` for this check:

1. Sign in to MongoDB Atlas and open the project containing the configured
   cluster.
2. Open the cluster's **Browse Collections** view.
3. Select database `INSURANCE`, then collection `INSURANCE_DATA`.
4. Open the collection's **Indexes** tab.
5. Confirm that an index has the key `id: 1` (ascending). The name may be
   `vehicle_insurance_id_ascending`, `id_1`, or another name; the key is what
   matters. The default `_id_` index does not satisfy this requirement.
6. If it is absent, choose **Create Index**, add field `id`, select ascending
   (`1`), leave **Unique** disabled, and use the name
   `vehicle_insurance_id_ascending`. Wait until Atlas shows the index as ready.

The same check is available in MongoDB Compass: open `INSURANCE` →
`INSURANCE_DATA` → **Indexes** and look for the same `id: 1` key. Do not add a
unique constraint unless the database owner has separately approved that
schema change; the pipeline validates ID uniqueness as data quality instead.

Before training, also confirm in Atlas that the cluster runs MongoDB 5.0 or
newer and is a replica set or sharded cluster. The pipeline requires MongoDB
snapshot sessions for a consistent read boundary and intentionally fails if
the server cannot provide them.

Because strict Git provenance runs before ingestion, commit the reviewed source,
configuration, and test changes before starting a real pipeline run. The personal
`daily log.md` and other private notes stay outside that commit.

## Retry and rerun behavior

There are three intentionally different cases:

1. A completed artifact directory is rerun: the existing CSV and manifest are
   verified and reused; MongoDB is not queried again.
2. The snapshot cursor fails after reading begins: extraction aborts. It is not
   retried from the beginning because the database may have changed. The
   artifact records the abort and requires a new artifact directory for a new
   extraction request.
3. A new extraction is intentionally requested with a new artifact directory:
   MongoDB is read in a new snapshot session and receives a new `snapshot_id`.

This makes a retry auditable: either the same immutable snapshot is reused, or
the pipeline clearly starts a new extraction. It never combines records from
two database states.

## Operational check

After a candidate-only run, inspect these files under the same artifact run:

```text
artifact/<run>/data_ingestion/feature_store/data.csv
artifact/<run>/data_ingestion/feature_store/source_manifest.json
artifact/<run>/model_trainer/metrics.json
artifact/<run>/model_evaluation/report.json
```

The `snapshot_id`, `content_hash`, source namespace, and read-boundary fields
should be identical wherever the source manifest is represented.
