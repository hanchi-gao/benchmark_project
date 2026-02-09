"""Tests for web UI configuration."""

from webui.config import CHART_TABS, X_AXIS_FIELD, X_AXIS_LABEL


class TestChartConfig:
    """Tests for chart configuration completeness."""

    def test_overview_tab_exists(self):
        assert "Overview" in CHART_TABS

    def test_overview_has_six_charts(self):
        overview = CHART_TABS["Overview"]
        assert overview["type"] == "overview"
        assert len(overview["charts"]) == 6

    def test_single_tabs_have_required_fields(self):
        for tab_name, tab_config in CHART_TABS.items():
            if tab_config["type"] == "single":
                assert "y_field" in tab_config, f"{tab_name} missing y_field"
                assert "y_label" in tab_config, f"{tab_name} missing y_label"
                assert "title" in tab_config, f"{tab_name} missing title"

    def test_x_axis_configured(self):
        assert X_AXIS_FIELD == "max_concurrent_requests"
        assert len(X_AXIS_LABEL) > 0
