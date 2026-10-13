import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import json
import os
import sys

from flarewatch.paths import RESULTS_CSV

class SafeStream:
    """Wrapper around stdout/stderr to prevent OSError (Errno 5 EIO) when terminal is disconnected."""
    def __init__(self, target):
        self.target = target

    def write(self, s):
        try:
            if self.target:
                return self.target.write(s)
        except (OSError, BrokenPipeError, IOError):
            pass

    def flush(self):
        try:
            if self.target:
                return self.target.flush()
        except (OSError, BrokenPipeError, IOError):
            pass

    def isatty(self):
        try:
            return self.target.isatty() if self.target else False
        except Exception:
            return False

    def __getattr__(self, name):
        return getattr(self.target, name)

if not isinstance(sys.stdout, SafeStream):
    sys.stdout = SafeStream(sys.stdout)
if not isinstance(sys.stderr, SafeStream):
    sys.stderr = SafeStream(sys.stderr)

def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except (OSError, BrokenPipeError, IOError):
        pass

# ---------------------------------------------------------------------------
# Flarewatch visual theme — Warm Charcoal & Sand (light + dark)
# ---------------------------------------------------------------------------
PALETTES = {
    "dark": {
        "bg": "#141311", "surface": "#211F1B", "surface2": "#292621",
        "border": "rgba(242,237,228,0.08)", "border_strong": "rgba(242,237,228,0.15)",
        "text": "#F2EDE4", "text2": "#BDB5A8", "muted": "#948D82",
        "sand": "#C7B69B", "sand_text": "#D3C4AB", "alert": "#B85C55", "alert_text": "#D98B83",
        "ok": "#93A08A", "stone": "#7D776E", "grid": "rgba(242,237,228,0.06)", "tooltip": "#2A2723",
        "resources": {
            'water': "#7F93A3", 'medical aid': "#B85C55", 'shelter': "#9A8AA0",
            'food': "#C9A15A", 'rescue': "#C08261", 'clothing': "#93A08A",
        },
    },
    "light": {
        "bg": "#F4F1EA", "surface": "#FCFBF8", "surface2": "#F3EFE8",
        "border": "rgba(20,19,17,0.09)", "border_strong": "rgba(20,19,17,0.16)",
        "text": "#1C1A17", "text2": "#4A453E", "muted": "#6F685E",
        "sand": "#A08A66", "sand_text": "#7A6648", "alert": "#A64B44", "alert_text": "#9A3F38",
        "ok": "#6E7D63", "stone": "#A39C91", "grid": "rgba(20,19,17,0.07)", "tooltip": "#FFFFFF",
        "resources": {
            'water': "#5F7484", 'medical aid': "#A64B44", 'shelter': "#7D6C84",
            'food': "#9C7A35", 'rescue': "#A4653F", 'clothing': "#6E7D63",
        },
    },
}

FLAREWATCH_MARK_SVG = (
    '<svg class="fw-mark" viewBox="0 0 48 48" aria-hidden="true">'
    '<rect x=".5" y=".5" width="47" height="47" rx="12" fill="#1F1D19" stroke="#C7B69B" stroke-opacity=".28"/>'
    '<g fill="none" stroke="#C7B69B" stroke-linecap="round" stroke-width="2.4">'
    '<path d="M13.35 18.54A13 13 0 0 0 13.35 33.46"/><path d="M34.65 18.54A13 13 0 0 1 34.65 33.46"/>'
    '<path d="M9.45 13.79A19 19 0 0 0 9.45 38.21" stroke-opacity=".4"/>'
    '<path d="M38.55 13.79A19 19 0 0 1 38.55 38.21" stroke-opacity=".4"/></g>'
    '<path fill="#B85C55" d="M24 8C26.5 13 33 17.5 33 26.5C33 31.5 29 35.5 24 35.5C19 35.5 15 31.5 15 26.5'
    'C15 22 17.5 19 19.5 16.5C20 19.5 21.5 21.5 23.5 22C22.5 17 23 12 24 8Z"/>'
    '<path fill="#F2EDE4" d="M24 21.5C26 24.5 28.5 25.8 28.5 28.7C28.5 31.2 26.5 33.2 24 33.2'
    'C21.5 33.2 19.5 31.2 19.5 28.7C19.5 26.2 22 24.8 24 21.5Z"/></svg>'
)


def active_palette():
    """Palette matching the theme the viewer has selected in Streamlit (dark if unknown)."""
    theme_type = None
    try:
        theme_type = st.context.theme.type
    except Exception:
        pass
    return PALETTES.get(theme_type or "dark", PALETTES["dark"])


def flarewatch_css(p):
    """Component styles for the custom HTML blocks, using the active palette."""
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
.stApp {{ font-family: 'Inter', sans-serif; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.4rem; max-width: 1320px; }}
.fw-header {{ display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap; padding:4px 0 22px; margin-bottom:22px; border-bottom:1px solid {p['border']}; }}
.fw-brand {{ display:flex; align-items:center; gap:14px; }}
.fw-mark {{ width:40px; height:40px; border-radius:10px; }}
.fw-word {{ font-family:'Instrument Serif',Georgia,serif; font-size:30px; line-height:1; color:{p['text']}; }}
.fw-tag {{ margin-top:4px; font-size:11px; font-weight:500; letter-spacing:.1em; text-transform:uppercase; color:{p['muted']}; }}
.fw-live {{ display:flex; align-items:center; gap:8px; height:28px; padding:0 12px; border-radius:999px; font-size:12px; color:{p['text2']}; border:1px solid {p['border']}; }}
.fw-live i {{ width:7px; height:7px; border-radius:50%; background:{p['ok']}; }}
.fw-kpis {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:1px; margin-bottom:18px; background:{p['border']};
  border:1px solid {p['border']}; border-radius:14px; overflow:hidden; }}
@media (max-width: 1100px) {{ .fw-kpis {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .fw-kpi:first-child {{ grid-column:1/-1; }} }}
.fw-kpi {{ background:{p['surface']}; padding:18px 20px 20px; transition:background-color .2s; }}
.fw-kpi:hover {{ background:{p['surface2']}; }}
.fw-kpi-label {{ font-size:12.5px; color:{p['muted']}; }}
.fw-kpi-value {{ font-family:'Instrument Serif',Georgia,serif; font-size:38px; line-height:1.05; margin-top:8px; color:{p['text']}; }}
.fw-kpi.alert .fw-kpi-value {{ color:{p['alert_text']}; }}
.fw-kpi-sub {{ margin-top:6px; font-size:12px; color:{p['muted']}; }}
.fw-section {{ margin:22px 0 10px; font-size:11.5px; font-weight:500; letter-spacing:.12em; text-transform:uppercase; color:{p['sand_text']}; }}
[data-testid="stPlotlyChart"] {{ background:{p['surface']}; border:1px solid {p['border']}; border-radius:14px; padding:6px 8px; }}
</style>
"""


def style_figure(fig, p, legend=False):
    """Apply the Flarewatch look to a Plotly figure."""
    fig.update_layout(
        template="plotly_white" if p is PALETTES["light"] else "plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=p["muted"], size=12),
        title_font=dict(family="Inter, sans-serif", size=14, color=p["text"]),
        margin=dict(l=10, r=28, t=52, b=10),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, x=0, font=dict(color=p["text2"])),
        hoverlabel=dict(bgcolor=p["tooltip"], bordercolor=p["border_strong"], font=dict(family="IBM Plex Mono, monospace", color=p["text"])),
    )
    fig.update_xaxes(gridcolor=p["grid"], zeroline=False, linecolor=p["grid"])
    fig.update_yaxes(gridcolor=p["grid"], zeroline=False, linecolor=p["grid"])
    return fig


class AutomatedDashboard:
    def __init__(self, results_file=RESULTS_CSV):
        self.results_file = results_file
        self.results_db = []
        self.load_existing_results()
    
    def load_existing_results(self):
        """Load existing results from file"""
        if os.path.exists(self.results_file):
            try:
                df = pd.read_csv(self.results_file)
                self.results_db = df.to_dict('records')
                safe_print(f"✅ Loaded {len(self.results_db)} existing results")
            except Exception as e:
                safe_print(f"❌ Error loading existing results: {e}")
    
    def update_dashboard(self, new_results):
        """Update dashboard with new classification results"""
        if new_results:
            self.results_db.extend(new_results)
            self.save_results()
            safe_print(f"✅ Updated dashboard with {len(new_results)} new results")
    
    def save_results(self):
        """Save all results to CSV file"""
        if self.results_db:
            df = pd.DataFrame(self.results_db)
            df.to_csv(self.results_file, index=False)
    
    def get_statistics(self):
        """Generate comprehensive statistics"""
        if not self.results_db:
            return {}
        
        df = pd.DataFrame(self.results_db)
        
        # Basic statistics
        stats = {
            'total_tweets': len(df),
            'requests_detected': len(df[df['is_request'] == True]),
            'normal_tweets': len(df[df['is_request'] == False]),
            'avg_confidence': df['confidence'].mean(),
            'high_confidence_requests': len(df[(df['is_request'] == True) & (df['confidence'] > 0.8)]),
            'low_confidence_requests': len(df[(df['is_request'] == True) & (df['confidence'] < 0.7)])
        }
        
        # Resource breakdown
        if 'resource' in df.columns:
            resource_df = df[df['is_request'] == True]
            if len(resource_df) > 0:
                stats['resource_breakdown'] = resource_df['resource'].value_counts().to_dict()
            else:
                stats['resource_breakdown'] = {}
        
        # Time-based statistics
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                stats['recent_requests'] = len(df[(df['is_request'] == True) & 
                                                 (df['timestamp'] > datetime.now() - timedelta(hours=1))])
                stats['requests_last_24h'] = len(df[(df['is_request'] == True) & 
                                                   (df['timestamp'] > datetime.now() - timedelta(hours=24))])
            except:
                stats['recent_requests'] = 0
                stats['requests_last_24h'] = 0
        
        return stats
    
    def create_streamlit_app(self):
        """Create the Flarewatch Streamlit dashboard"""
        try:
            st.set_page_config(page_title="Flarewatch · Crisis Signal Intelligence", page_icon="🔥", layout="wide")
        except Exception:
            pass

        p = active_palette()
        st.markdown(flarewatch_css(p), unsafe_allow_html=True)

        stats = self.get_statistics()

        # Single-line HTML: indented lines would be parsed as a Markdown code block
        st.markdown(
            '<div class="fw-header"><div class="fw-brand">' + FLAREWATCH_MARK_SVG +
            '<div><div class="fw-word">Flarewatch</div>'
            '<div class="fw-tag">Signal Intelligence · Monitoring Console</div></div></div>'
            f'<div class="fw-live"><i></i>Refreshed {datetime.now().strftime("%H:%M:%S")}</div></div>',
            unsafe_allow_html=True,
        )

        if not stats:
            st.warning("No data available. Please run the classifier first.")
            return

        total = stats['total_tweets']
        requests = stats['requests_detected']
        request_pct = (requests / total * 100) if total else 0.0
        high_pct = (stats['high_confidence_requests'] / requests * 100) if requests else 0.0

        cards = [
            ("Signals analyzed", f"{total:,}", "Total classified telemetry", ""),
            ("Disaster requests", f"{requests:,}", f"{request_pct:.1f}% of volume", "alert"),
            ("Normal signals", f"{stats['normal_tweets']:,}", f"{100 - request_pct:.1f}% non-critical", ""),
            ("Mean P(request)", f"{stats['avg_confidence']:.3f}", "Across all signals", ""),
            ("High confidence (>80%)", f"{stats['high_confidence_requests']:,}", f"{high_pct:.1f}% of requests", ""),
        ]
        st.markdown(
            '<div class="fw-kpis">' + "".join(
                f'<div class="fw-kpi {accent}"><div class="fw-kpi-label">{label}</div>'
                f'<div class="fw-kpi-value">{value}</div><div class="fw-kpi-sub">{sub}</div></div>'
                for label, value, sub, accent in cards
            ) + '</div>',
            unsafe_allow_html=True,
        )

        df = pd.DataFrame(self.results_db)
        tab_overview, tab_signals, tab_export = st.tabs(["Overview", "Signal stream", "Export"])

        with tab_overview:
            col1, col2 = st.columns(2)

            with col1:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['Disaster Requests', 'Normal Tweets'],
                    values=[requests, stats['normal_tweets']],
                    hole=0.72,
                    marker=dict(colors=[p['alert'], p['sand']], line=dict(color=p['surface'], width=3)),
                    textinfo='none',
                    hovertemplate='%{label}<br>%{value:,} signals (%{percent})<extra></extra>',
                    sort=False,
                )])
                fig_pie.add_annotation(text=f"{request_pct:.1f}%<br><span style='font-size:11px;color:{p['muted']}'>disaster</span>",
                                       showarrow=False, font=dict(family="Instrument Serif, Georgia, serif", size=30, color=p['text']))
                fig_pie.update_layout(title="Request vs Normal Distribution")
                st.plotly_chart(style_figure(fig_pie, p, legend=True), use_container_width=True, theme=None)

            with col2:
                if stats.get('resource_breakdown'):
                    breakdown = sorted(stats['resource_breakdown'].items(), key=lambda kv: kv[1])
                    resources = [k for k, _ in breakdown]
                    counts = [v for _, v in breakdown]
                    fig_bar = go.Figure(go.Bar(
                        x=counts, y=resources, orientation='h',
                        marker=dict(color=[p['resources'].get(r, p['stone']) for r in resources], line=dict(width=0)),
                        hovertemplate='%{y}: %{x:,} requests<extra></extra>',
                    ))
                    fig_bar.update_layout(title="Resource Type Breakdown", xaxis_title="Requests", yaxis_title=None)
                    st.plotly_chart(style_figure(fig_bar, p), use_container_width=True, theme=None)
                else:
                    st.info("No resource breakdown data available")

            if 'confidence' in df.columns:
                fig_hist = go.Figure()
                for is_req, name, color in [(True, 'Disaster requests', p['alert']), (False, 'Normal tweets', p['sand'])]:
                    subset = df[df['is_request'] == is_req]['confidence']
                    fig_hist.add_trace(go.Histogram(x=subset, name=name, marker_color=color, opacity=0.85,
                                                    xbins=dict(start=0, end=1, size=0.05)))
                fig_hist.add_vline(x=0.5, line_dash="dot", line_color=p['muted'],
                                   annotation_text="threshold 0.50", annotation_font_color=p['muted'])
                fig_hist.update_layout(title="Confidence Score Distribution", barmode='overlay',
                                       xaxis_title="P(disaster)", yaxis_title="Signals")
                st.plotly_chart(style_figure(fig_hist, p, legend=True), use_container_width=True, theme=None)

            # Recent activity
            st.markdown('<div class="fw-section">Recent activity</div>', unsafe_allow_html=True)
            if 'timestamp' in df.columns:
                try:
                    ts_df = df.copy()
                    ts_df['timestamp'] = pd.to_datetime(ts_df['timestamp'])
                    recent_df = ts_df[ts_df['timestamp'] > datetime.now() - timedelta(hours=24)]

                    if len(recent_df) > 0:
                        recent_df = recent_df.assign(hour=recent_df['timestamp'].dt.hour)
                        hourly_requests = recent_df[recent_df['is_request'] == True].groupby('hour').size()

                        fig_time = go.Figure(go.Scatter(
                            x=hourly_requests.index, y=hourly_requests.values, mode='lines+markers',
                            line=dict(color=p['sand'], width=2, shape='spline'),
                            fill='tozeroy', fillcolor='rgba(199,182,155,0.12)',
                            marker=dict(size=7, color=p['surface'], line=dict(color=p['sand'], width=2)),
                        ))
                        fig_time.update_layout(title="Requests per Hour (Last 24 Hours)", xaxis_title="Hour", yaxis_title="Requests")
                        st.plotly_chart(style_figure(fig_time, p), use_container_width=True, theme=None)
                    else:
                        st.info("No recent activity in the last 24 hours")
                except Exception:
                    st.info("Could not parse timestamp data")

        with tab_signals:
            st.markdown('<div class="fw-section">Recent classification results</div>', unsafe_allow_html=True)
            # Show random sample of up to 25 results
            sample_size = min(25, len(df))
            recent_results = df.sample(n=sample_size) if sample_size > 0 else df

            display_df = recent_results[['tweet', 'is_request', 'confidence', 'resource']].copy()
            display_df['related_to_disaster'] = display_df['is_request'].apply(lambda x: "Yes" if x else "No")
            display_df = display_df[['tweet', 'related_to_disaster', 'confidence', 'resource']]
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'tweet': st.column_config.TextColumn("Signal", width="large"),
                    'related_to_disaster': st.column_config.TextColumn("Disaster?"),
                    'confidence': st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="%.3f"),
                    'resource': st.column_config.TextColumn("Resource"),
                },
            )
            st.button("Shuffle sample")

        with tab_export:
            st.markdown('<div class="fw-section">Export data</div>', unsafe_allow_html=True)
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download results (CSV)",
                    data=df.to_csv(index=False),
                    file_name=f"disaster_classification_results_{stamp}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col2:
                st.download_button(
                    label="Download statistics (JSON)",
                    data=json.dumps(stats, indent=2, default=str),
                    file_name=f"classification_statistics_{stamp}.json",
                    mime="application/json",
                    use_container_width=True,
                )

    def generate_report(self, filename="dashboard_report.html"):
        """Generate an HTML report"""
        stats = self.get_statistics()
        
        if not stats:
            return "No data available for report"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Disaster Response Classification Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 20px; padding: 15px; background-color: #e8f4fd; border-radius: 5px; }}
                .chart {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Disaster Response Classification Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h2>Summary Statistics</h2>
            <div class="metric">
                <strong>Total Tweets:</strong> {stats['total_tweets']}
            </div>
            <div class="metric">
                <strong>Requests Detected:</strong> {stats['requests_detected']}
            </div>
            <div class="metric">
                <strong>Average Confidence:</strong> {stats['avg_confidence']:.3f}
            </div>
            <div class="metric">
                <strong>High Confidence Requests:</strong> {stats['high_confidence_requests']}
            </div>
        """
        
        if stats.get('resource_breakdown'):
            html_content += """
            <h2>Resource Breakdown</h2>
            <table>
                <tr><th>Resource Type</th><th>Count</th></tr>
            """
            for resource, count in stats['resource_breakdown'].items():
                html_content += f"<tr><td>{resource}</td><td>{count}</td></tr>"
            html_content += "</table>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(filename, 'w') as f:
            f.write(html_content)
        
        safe_print(f"Report generated: {filename}")
        return filename
    
    def print_summary(self):
        """Print a text summary to console"""
        stats = self.get_statistics()
        
        if not stats:
            safe_print("No data available")
            return
        
        safe_print("\n" + "="*60)
        safe_print("DASHBOARD SUMMARY")
        safe_print("="*60)
        safe_print(f"Total tweets processed: {stats['total_tweets']}")
        safe_print(f"Disaster requests detected: {stats['requests_detected']}")
        safe_print(f"Normal tweets: {stats['normal_tweets']}")
        safe_print(f"Average confidence: {stats['avg_confidence']:.3f}")
        safe_print(f"High confidence requests (>80%): {stats['high_confidence_requests']}")
        safe_print(f"Low confidence requests (<70%): {stats['low_confidence_requests']}")
        
        if stats.get('resource_breakdown'):
            safe_print("\nResource Breakdown:")
            for resource, count in stats['resource_breakdown'].items():
                safe_print(f"  {resource}: {count}")
        
        safe_print("="*60) 