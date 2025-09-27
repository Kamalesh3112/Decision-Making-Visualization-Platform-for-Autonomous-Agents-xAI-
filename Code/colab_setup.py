# =====================================================================
# GOOGLE COLAB SETUP AND EXECUTION GUIDE
# =====================================================================

# 1. INSTALL REQUIRED PACKAGES IN COLAB
# Run this cell first in Google Colab:

!pip install streamlit pandas numpy scikit-learn plotly seaborn matplotlib umap-learn networkx scipy
!pip install pyngrok  # For tunneling Streamlit in Colab
!npm install localtunnel  # Alternative tunneling option

# 2. SETUP NGROK FOR STREAMLIT (Optional - for dashboard)
# Get your free ngrok token from https://ngrok.com/
from pyngrok import ngrok

# Set your ngrok auth token (replace with your actual token)
# ngrok.set_auth_token("YOUR_NGROK_TOKEN_HERE")

# 3. COLAB-OPTIMIZED VERSION WITH JUPYTER WIDGETS
# This version works better in Colab environment

import warnings
warnings.filterwarnings('ignore')

# Import all the main code from the original platform
# (Copy the entire xAI platform code here or import it)

# =====================================================================
# COLAB-SPECIFIC MODIFICATIONS
# =====================================================================

def run_colab_analysis():
    """Optimized analysis for Google Colab environment"""
    print("🤖 Decision-Making Visualization Platform - Google Colab Version")
    print("=" * 70)
    
    # Initialize components
    print("🔄 Initializing system components...")
    simulator = AgentSimulator(num_agents=25)
    db_manager = DatabaseManager("colab_agent_decisions.db")
    analyzer = UnsupervisedAnalyzer()
    fairness_analyzer = FairnessAnalyzer()
    viz_engine = VisualizationEngine()
    
    # Generate sample data
    print("📊 Generating sample decision data...")
    decisions = simulator.generate_batch_decisions(800)
    db_manager.store_decisions(decisions)
    
    # Load and analyze data
    print("🔍 Loading and analyzing data...")
    df = db_manager.load_decisions_as_dataframe()
    
    print(f"\n📈 Dataset Overview:")
    print(f"- Total decisions: {len(df)}")
    print(f"- Unique agents: {df['agent_id'].nunique()}")
    print(f"- Agent types: {', '.join(df['agent_type'].unique())}")
    print(f"- Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Display basic statistics
    display(df.describe())
    
    # Run unsupervised analysis
    print("\n🎯 Running Unsupervised ML Analysis...")
    cluster_results = analyzer.detect_bias_clusters(df)
    reduction_results = analyzer.dimensionality_reduction(df)
    anomalies = analyzer.detect_anomalies(df)
    
    print(f"✅ Analysis Results:")
    print(f"- K-Means clusters found: {cluster_results['kmeans']['n_clusters']}")
    print(f"- Silhouette score: {cluster_results['kmeans']['silhouette_score']:.3f}")
    print(f"- DBSCAN clusters: {cluster_results['dbscan']['n_clusters']}")
    print(f"- Anomalous decisions: {np.sum(anomalies == -1)}")
    
    # Create and display visualizations
    print("\n📊 Generating Visualizations...")
    
    # 1. Bias Cluster Visualization
    fig_clusters = viz_engine.plot_bias_clusters(df, cluster_results, reduction_results)
    fig_clusters.show()
    
    # 2. Fairness Analysis
    group_analysis = fairness_analyzer.analyze_group_fairness(df)
    fig_fairness = viz_engine.plot_fairness_metrics(group_analysis)
    fig_fairness.show()
    
    # 3. Temporal Analysis
    fig_temporal = viz_engine.plot_temporal_analysis(df)
    fig_temporal.show()
    
    # Generate alerts and recommendations
    print("\n🚨 Fairness Analysis:")
    alerts = fairness_analyzer.generate_fairness_alerts(df)
    
    if alerts:
        print(f"⚠️  Found {len(alerts)} active alerts:")
        for i, alert in enumerate(alerts[:5]):  # Show first 5 alerts
            print(f"{i+1}. {alert['type'].replace('_', ' ').title()}")
            print(f"   Severity: {alert['severity']}")
            print(f"   Recommendation: {alert['recommendation']}")
            print()
    else:
        print("✅ No active alerts - system operating within fairness thresholds!")
    
    # Summary metrics
    print("📈 Final System Metrics:")
    avg_bias = df['bias_algorithmic_bias'].mean()
    avg_fairness = df[[col for col in df.columns if col.startswith('fairness_')]].mean().mean()
    avg_confidence = df['confidence_score'].mean()
    
    print(f"- Average algorithmic bias: {avg_bias:.3f}")
    print(f"- Average fairness score: {avg_fairness:.3f}")
    print(f"- Average confidence: {avg_confidence:.3f}")
    
    return df, cluster_results, group_analysis, alerts

def run_streamlit_in_colab():
    """Run Streamlit dashboard in Google Colab using ngrok"""
    
    # Write the main code to a file
    with open('xai_platform.py', 'w') as f:
        f.write('''
# Main xAI Platform code goes here
# (Copy the entire platform code from the original artifact)
        ''')
    
    # Start Streamlit in background
    import subprocess
    import threading
    
    def run_streamlit():
        subprocess.run(['streamlit', 'run', 'xai_platform.py', '--server.port', '8501'])
    
    # Start Streamlit in a separate thread
    streamlit_thread = threading.Thread(target=run_streamlit)
    streamlit_thread.daemon = True
    streamlit_thread.start()
    
    # Create ngrok tunnel
    from pyngrok import ngrok
    
    # Open ngrok tunnel to port 8501
    public_url = ngrok.connect(8501)
    print(f"🌐 Streamlit Dashboard URL: {public_url}")
    print("Click the link above to access your dashboard!")
    
    return public_url

# =====================================================================
# COLAB INTERACTIVE WIDGETS VERSION
# =====================================================================

from IPython.display import display, HTML
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual

def create_colab_interactive_dashboard():
    """Create an interactive dashboard using Jupyter widgets"""
    
    print("🤖 Interactive xAI Dashboard for Google Colab")
    print("=" * 50)
    
    # Initialize data
    simulator = AgentSimulator(num_agents=20)
    db_manager = DatabaseManager("interactive_decisions.db")
    decisions = simulator.generate_batch_decisions(500)
    db_manager.store_decisions(decisions)
    df = db_manager.load_decisions_as_dataframe()
    
    # Create widgets
    agent_type_widget = widgets.SelectMultiple(
        options=list(df['agent_type'].unique()),
        value=list(df['agent_type'].unique()),
        description='Agent Types:',
        disabled=False
    )
    
    context_widget = widgets.SelectMultiple(
        options=list(df['context'].unique()),
        value=list(df['context'].unique()),
        description='Contexts:',
        disabled=False
    )
    
    analysis_type_widget = widgets.Dropdown(
        options=[
            ('Bias Clusters', 'bias'),
            ('Fairness Analysis', 'fairness'),
            ('Temporal Analysis', 'temporal'),
            ('Overview Statistics', 'overview')
        ],
        value='overview',
        description='Analysis Type:',
    )
    
    def update_analysis(agent_types, contexts, analysis_type):
        # Filter data
        filtered_df = df[
            (df['agent_type'].isin(agent_types)) &
            (df['context'].isin(contexts))
        ]
        
        if len(filtered_df) == 0:
            print("No data matching the selected filters.")
            return
        
        print(f"📊 Filtered Dataset: {len(filtered_df)} decisions")
        
        if analysis_type == 'overview':
            display(filtered_df.describe())
            
            # Quick metrics
            avg_bias = filtered_df['bias_algorithmic_bias'].mean()
            avg_fairness = filtered_df[[col for col in filtered_df.columns if col.startswith('fairness_')]].mean().mean()
            
            print(f"\n📈 Key Metrics:")
            print(f"- Average Bias: {avg_bias:.3f}")
            print(f"- Average Fairness: {avg_fairness:.3f}")
            print(f"- Anomalous Decisions: {len(filtered_df[filtered_df['confidence_score'] < 0.3])}")
            
        elif analysis_type == 'bias':
            analyzer = UnsupervisedAnalyzer()
            cluster_results = analyzer.detect_bias_clusters(filtered_df)
            reduction_results = analyzer.dimensionality_reduction(filtered_df)
            
            viz_engine = VisualizationEngine()
            fig = viz_engine.plot_bias_clusters(filtered_df, cluster_results, reduction_results)
            fig.show()
            
        elif analysis_type == 'fairness':
            fairness_analyzer = FairnessAnalyzer()
            group_analysis = fairness_analyzer.analyze_group_fairness(filtered_df)
            
            viz_engine = VisualizationEngine()
            fig = viz_engine.plot_fairness_metrics(group_analysis)
            fig.show()
            
        elif analysis_type == 'temporal':
            viz_engine = VisualizationEngine()
            fig = viz_engine.plot_temporal_analysis(filtered_df)
            fig.show()
    
    # Create interactive interface
    interactive_dashboard = interactive(
        update_analysis,
        agent_types=agent_type_widget,
        contexts=context_widget,
        analysis_type=analysis_type_widget
    )
    
    display(interactive_dashboard)

# =====================================================================
# MAIN EXECUTION FOR COLAB
# =====================================================================

def main_colab():
    """Main function for Google Colab execution"""
    
    print("🚀 Choose your execution mode:")
    print("1. Basic Analysis (Recommended for beginners)")
    print("2. Interactive Widgets Dashboard")
    print("3. Full Streamlit Dashboard (requires ngrok setup)")
    
    mode = input("Enter choice (1/2/3): ").strip()
    
    if mode == "1":
        # Run basic analysis
        df, cluster_results, group_analysis, alerts = run_colab_analysis()
        print("✅ Analysis complete! Check the visualizations above.")
        
    elif mode == "2":
        # Run interactive widgets version
        create_colab_interactive_dashboard()
        
    elif mode == "3":
        # Run full Streamlit dashboard
        print("⚠️  Make sure you have set up ngrok authentication first!")
        confirm = input("Continue with Streamlit setup? (y/n): ").strip().lower()
        if confirm == 'y':
            url = run_streamlit_in_colab()
            print(f"Dashboard available at: {url}")
        else:
            print("Streamlit setup cancelled.")
    
    else:
        print("Invalid choice. Running basic analysis...")
        run_colab_analysis()

# Execute main function
if __name__ == "__main__":
    main_colab()

# =====================================================================
# COLAB INSTALLATION AND SETUP INSTRUCTIONS
# =====================================================================

"""
GOOGLE COLAB SETUP INSTRUCTIONS:
===============================

1. Create a new Google Colab notebook
2. Copy and paste this entire code
3. Run the installation cell first:
   !pip install streamlit pandas numpy scikit-learn plotly seaborn matplotlib umap-learn networkx scipy

4. For full dashboard with ngrok:
   - Sign up at https://ngrok.com/
   - Get your auth token
   - Run: !pip install pyngrok
   - Set token: ngrok.set_auth_token("YOUR_TOKEN")

5. Execute the main code

FEATURES IN COLAB:
=================
✅ Full ML analysis pipeline
✅ Interactive visualizations with Plotly
✅ Jupyter widgets for dynamic filtering
✅ Streamlit dashboard via ngrok tunnel
✅ All original functionality preserved
✅ Colab-optimized display methods

LIMITATIONS:
===========
- Streamlit requires ngrok for external access
- Large datasets may be slower than local execution
- Some advanced features may need GPU runtime
"""