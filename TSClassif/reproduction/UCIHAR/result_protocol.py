"""Pure helpers for interpreting training-reported metric provenance."""

METRIC_PROTOCOLS = {
    "official_stateful_no_reset",
    "stateless_current",
    "clean_checkpoint",
}


def reported_metric_fields(row):
    protocol = row.get("metric_protocol", "")
    if protocol == "official_stateful_no_reset":
        return {
            "official_reported_f1": row.get("f1_score", ""),
            "current_reported_f1": "",
        }
    return {
        "official_reported_f1": "",
        "current_reported_f1": row.get("f1_score", ""),
    }

