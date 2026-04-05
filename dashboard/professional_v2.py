import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="MedFL - Federated Learning Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== SOPHISTICATED CSS (Soft Colors, Clear Contrast) ==========
st.markdown("""
<style>
    /* Main background - Soft warm gray */
    .stApp {
        background: linear-gradient(135deg, #eef2f3 0%, #e0e7eb 100%);
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2a3a 0%, #0f1a24 100%);
    }
    
    /* Professional card with subtle depth */
    .card {
        background: #ffffff;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.02);
        border: 1px solid rgba(0,0,0,0.06);
    }
    
    /* Metric cards with soft backgrounds */
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 16px;
        border-left: 3px solid;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 600;
        color: #1e293b;
        margin: 8px 0;
    }
    
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #64748b;
    }
    
    /* Tab styling - FIXED VISIBILITY */
    .stTabs {
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #ffffff;
        padding: 8px 12px;
        border-radius: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 0 24px;
        border-radius: 10px;
        font-size: 14px;
        font-weight: 500;
        color: #475569;
        background: transparent;
        transition: all 0.2s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #f1f5f9;
        color: #1e293b;
    }
    
    .stTabs [aria-selected="true"] {
        background: #1e293b;
        color: #ffffff;
    }
    
    /* Headline colors for markdown headers */
    h1, h2, h3, h4, h5, h6 {
        color: #1e293b !important;
        font-weight: 600;
    }
    
    h3 {
        color: #1e293b !important;
        font-size: 18px;
        margin-bottom: 16px;
    }
    
    /* Hospital cards with soft colored borders */
    .hospital-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 18px;
        margin: 10px 0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .hospital-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .hospital-a { border-left: 4px solid #3b82f6; }
    .hospital-b { border-left: 4px solid #10b981; }
    .hospital-c { border-left: 4px solid #8b5cf6; }
    
    .hospital-title {
        font-size: 16px;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 12px;
    }
    
    .hospital-stat {
        font-size: 13px;
        color: #475569;
        margin: 6px 0;
    }
    
    .hospital-value {
        font-weight: 600;
        color: #0f172a;
    }
    
    /* Progress bar */
    .progress-track {
        background: #e2e8f0;
        border-radius: 20px;
        height: 6px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .progress-fill-blue { background: #3b82f6; width: 85.2%; height: 100%; border-radius: 20px; }
    .progress-fill-green { background: #10b981; width: 84.7%; height: 100%; border-radius: 20px; }
    .progress-fill-purple { background: #8b5cf6; width: 87.3%; height: 100%; border-radius: 20px; }
    
    /* Header section */
    .header-section {
        background: #ffffff;
        border-radius: 20px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        border: 1px solid #e2e8f0;
    }
    
    .page-title {
        font-size: 28px;
        font-weight: 700;
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .page-subtitle {
        font-size: 14px;
        color: #64748b;
        margin-top: 8px;
    }
    
    /* Info panels */
    .info-panel {
        background: #f8fafc;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #e2e8f0;
    }
    
    .info-panel-blue { background: #eff6ff; border-left: 3px solid #3b82f6; }
    .info-panel-green { background: #ecfdf5; border-left: 3px solid #10b981; }
    .info-panel-purple { background: #f5f3ff; border-left: 3px solid #8b5cf6; }
    
    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
        margin: 20px 0;
    }
    
    /* Footer */
    .footer {
        background: #ffffff;
        border-radius: 12px;
        padding: 16px 24px;
        margin-top: 24px;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ========== HEADER SECTION ==========
st.markdown("""
<div class="header-section">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
            <h1 class="page-title">🏥 MedFL</h1>
            <p class="page-subtitle">Federated Learning Platform for Medical Imaging | Privacy-Preserving Collaborative Training</p>
        </div>
        <div style="background: #f1f5f9; padding: 8px 16px; border-radius: 40px;">
            <span style="color: #10b981; font-size: 12px;">●</span>
            <span style="color: #475569; font-size: 12px; margin-left: 6px;">System Online</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ========== METRIC ROW ==========
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card" style="border-left-color: #3b82f6;">
        <div class="metric-label">ACTIVE HOSPITALS</div>
        <div class="metric-value">3 / 3</div>
        <div style="color: #10b981; font-size: 11px;">✓ All connected</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card" style="border-left-color: #10b981;">
        <div class="metric-label">TRAINING ROUNDS</div>
        <div class="metric-value">10 / 10</div>
        <div style="color: #64748b; font-size: 11px;">✓ Complete</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card" style="border-left-color: #8b5cf6;">
        <div class="metric-label">GLOBAL ACCURACY</div>
        <div class="metric-value">87.3%</div>
        <div style="color: #3b82f6; font-size: 11px;">↑ +2.4% improvement</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card" style="border-left-color: #ef4444;">
        <div class="metric-label">PRIVACY BUDGET (ε)</div>
        <div class="metric-value">2.8</div>
        <div style="color: #10b981; font-size: 11px;">✓ HIPAA compliant</div>
    </div>
    """, unsafe_allow_html=True)

# ========== TABS WITH BETTER VISIBILITY ==========
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Training Analytics", 
    "🏥  Hospital Insights", 
    "🔒  Privacy & Security", 
    "🎯  Model Explainability"
])

# ========== TAB 1: TRAINING ANALYTICS ==========
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Training Progress Over Communication Rounds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        rounds = list(range(1, 11))
        accuracies = [35.2, 42.8, 49.5, 56.3, 63.1, 69.8, 75.4, 81.2, 85.7, 87.3]
        
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            x=rounds, y=accuracies,
            mode='lines+markers',  # FIXED: was 'lines+prors'
            name='Global Accuracy',
            line=dict(color='#3b82f6', width=2.5),
            marker=dict(size=8, color='#3b82f6'),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.08)'
        ))
        
        fig_acc.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_title="Round",
            yaxis_title="Accuracy (%)",
            yaxis_range=[20, 100],
            font=dict(family="Inter, sans-serif", size=12, color="#334155"),
            xaxis=dict(showgrid=True, gridcolor='#e2e8f0'),
            yaxis=dict(showgrid=True, gridcolor='#e2e8f0'),
            height=350,
            margin=dict(l=40, r=20, t=20, b=40)
        )
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with col2:
        losses = [1.85, 1.62, 1.41, 1.23, 1.08, 0.94, 0.81, 0.68, 0.54, 0.42]
        
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=rounds, y=losses,
            mode='lines+markers',  # FIXED: was 'lines+prors'
            name='Training Loss',
            line=dict(color='#ef4444', width=2.5),
            marker=dict(size=8, color='#ef4444'),
            fill='tozeroy',
            fillcolor='rgba(239, 68, 68, 0.08)'
        ))
        
        fig_loss.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_title="Round",
            yaxis_title="Loss",
            font=dict(family="Inter, sans-serif", size=12, color="#334155"),
            xaxis=dict(showgrid=True, gridcolor='#e2e8f0'),
            yaxis=dict(showgrid=True, gridcolor='#e2e8f0'),
            height=350,
            margin=dict(l=40, r=20, t=20, b=40)
        )
        st.plotly_chart(fig_loss, use_container_width=True)
    
    # Summary table
    st.markdown("### Round-by-Round Summary")
    summary_df = pd.DataFrame({
        'Round': rounds,
        'Accuracy': [f"{a:.1f}%" for a in accuracies],
        'Loss': [f"{l:.2f}" for l in losses],
        'Δ Accuracy': ['-'] + [f"+{accuracies[i]-accuracies[i-1]:.1f}%" for i in range(1, len(accuracies))]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========== TAB 2: HOSPITAL INSIGHTS ==========
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Individual Hospital Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="hospital-card hospital-a">
            <div class="hospital-title">🏥 Hospital A</div>
            <div class="hospital-stat">📊 Training Samples: <span class="hospital-value">930</span></div>
            <div class="hospital-stat">🎯 Final Accuracy: <span class="hospital-value" style="color: #3b82f6;">85.2%</span></div>
            <div class="hospital-stat">⏱️ Training Time: <span class="hospital-value">47 min</span></div>
            <div class="progress-track"><div class="progress-fill-blue"></div></div>
            <div class="hospital-stat" style="font-size: 11px;">🏥 Specialty: General Radiology</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="hospital-card hospital-b">
            <div class="hospital-title">🏥 Hospital B</div>
            <div class="hospital-stat">📊 Training Samples: <span class="hospital-value">1,044</span></div>
            <div class="hospital-stat">🎯 Final Accuracy: <span class="hospital-value" style="color: #10b981;">84.7%</span></div>
            <div class="hospital-stat">⏱️ Training Time: <span class="hospital-value">52 min</span></div>
            <div class="progress-track"><div class="progress-fill-green"></div></div>
            <div class="hospital-stat" style="font-size: 11px;">🏥 Specialty: Pediatric Radiology</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="hospital-card hospital-c">
            <div class="hospital-title">🏥 Hospital C</div>
            <div class="hospital-stat">📊 Training Samples: <span class="hospital-value">497</span></div>
            <div class="hospital-stat">🎯 Final Accuracy: <span class="hospital-value" style="color: #8b5cf6;">87.3%</span></div>
            <div class="hospital-stat">⏱️ Training Time: <span class="hospital-value">25 min</span></div>
            <div class="progress-track"><div class="progress-fill-purple"></div></div>
            <div class="hospital-stat" style="font-size: 11px;">🏥 Specialty: Pulmonary Medicine</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_bar = go.Figure(data=[
            go.Bar(
                x=['Hospital A', 'Hospital B', 'Hospital C'],
                y=[85.2, 84.7, 87.3],
                marker_color=['#3b82f6', '#10b981', '#8b5cf6'],
                text=[85.2, 84.7, 87.3],
                textposition='outside'
            )
        ])
        fig_bar.update_layout(
            title="Accuracy by Hospital",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color="#334155"),
            height=350,
            yaxis_range=[70, 95]
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        fig_pie = go.Figure(data=[
            go.Pie(
                labels=['Hospital A (930)', 'Hospital B (1,044)', 'Hospital C (497)'],
                values=[930, 1044, 497],
                marker_colors=['#3b82f6', '#10b981', '#8b5cf6'],
                hole=0.4,
                textinfo='percent'
            )
        ])
        fig_pie.update_layout(
            title="Data Distribution",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color="#334155"),
            height=350
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========== TAB 3: PRIVACY & SECURITY ==========
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-panel info-panel-blue">
            <h4 style="color: #1e293b; margin: 0 0 12px 0;">🔒 HIPAA Compliance Status</h4>
            <p style="color: #475569; font-size: 13px;">✓ Differential Privacy Enabled (ε=2.8)</p>
            <p style="color: #475569; font-size: 13px;">✓ No Patient Data Shared Between Hospitals</p>
            <p style="color: #475569; font-size: 13px;">✓ Secure Aggregation Protocol Active</p>
            <p style="color: #475569; font-size: 13px;">✓ Audit Trail Available for Review</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-panel info-panel-green">
            <h4 style="color: #1e293b; margin: 0 0 12px 0;">📊 Privacy Budget Status</h4>
            <p style="color: #475569; font-size: 13px;">Budget Used: <strong>28%</strong> (ε=2.8 / 10.0)</p>
            <div class="progress-track" style="background: #e2e8f0;">
                <div style="width: 28%; background: #10b981; height: 100%; border-radius: 20px;"></div>
            </div>
            <p style="color: #475569; font-size: 13px; margin-top: 12px;">Remaining Budget: ε=7.2</p>
            <p style="color: #64748b; font-size: 11px;">Estimated rounds remaining: 25</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### Privacy vs. Accuracy Trade-off")
    
    tradeoff = pd.DataFrame({
        'Privacy Level': ['No Privacy', 'Low (ε=10)', 'Medium (ε=3)', 'High (ε=1)'],
        'Accuracy (%)': [89.2, 88.5, 87.3, 82.1]
    })
    
    fig_trade = px.line(
        tradeoff, x='Privacy Level', y='Accuracy (%)',
        markers=True, color_discrete_sequence=['#ef4444']
    )
    fig_trade.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color="#334155"),
        height=350
    )
    st.plotly_chart(fig_trade, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========== TAB 4: MODEL EXPLAINABILITY ==========
with tab4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-panel" style="background: #f8fafc;">
            <h4 style="color: #1e293b;">🩻 Chest X-ray Analysis</h4>
            <div style="background: #e2e8f0; border-radius: 12px; height: 200px; display: flex; align-items: center; justify-content: center; margin: 16px 0;">
                <div style="text-align: center;">
                    <div style="font-size: 64px;">🫁</div>
                    <p style="color: #475569;">Sample Chest X-ray</p>
                </div>
            </div>
            <div class="info-panel-blue" style="background: #eff6ff; border-radius: 8px; padding: 12px;">
                <p style="color: #1e3a8a; margin: 0;"><strong>Prediction:</strong> Pneumonia</p>
                <p style="color: #1e3a8a; margin: 4px 0 0 0;"><strong>Confidence:</strong> 87.3%</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-panel" style="background: #f8fafc;">
            <h4 style="color: #1e293b;">🔥 Grad-CAM Explanation</h4>
            <div style="background: #e2e8f0; border-radius: 12px; height: 200px; display: flex; align-items: center; justify-content: center; margin: 16px 0;">
                <div style="text-align: center;">
                    <div style="font-size: 64px;">🔥</div>
                    <p style="color: #475569;">Attention Heatmap Overlay</p>
                </div>
            </div>
            <div class="info-panel" style="background: #fef2f2; border-radius: 8px; padding: 12px;">
                <p style="color: #991b1b; margin: 0;"><strong>Clinical Insight:</strong> Model focuses on lower lung regions</p>
                <p style="color: #991b1b; margin: 4px 0 0 0; font-size: 11px;">✓ Consistent with medical literature</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature importance
    st.markdown("### Top Predictive Clinical Features")
    features = ['Lower Lung Opacity', 'Interstitial Markings', 'Air Bronchograms', 'Pleural Effusion', 'Cavitation']
    importance = [0.42, 0.28, 0.15, 0.09, 0.06]
    
    fig_feat = go.Figure(data=[
        go.Bar(
            x=importance, y=features, orientation='h',
            marker_color='#3b82f6',
            text=[f"{i*100:.0f}%" for i in importance],
            textposition='outside'
        )
    ])
    fig_feat.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(color="#334155"), height=300,
        xaxis_title="Importance Score", yaxis_title="Clinical Feature",
        margin=dict(l=120)
    )
    st.plotly_chart(fig_feat, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========== FOOTER ==========
st.markdown("""
<div class="footer">
    <p style="color: #64748b; font-size: 12px; margin: 0;">
        MedFL — Federated Learning Platform | HIPAA Compliant | Real-time Privacy Monitoring
    </p>
    <p style="color: #94a3b8; font-size: 11px; margin: 8px 0 0 0;">
        3 Active Hospitals | 2,471 Patient Cases Processed | Last updated: April 2026
    </p>
</div>
""", unsafe_allow_html=True)