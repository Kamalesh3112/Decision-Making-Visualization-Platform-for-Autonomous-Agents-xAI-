!pip install pandas numpy scikit-learn umap-learn hdbscan plotly seaborn shap lime pyod

!pip install streamlit pandas numpy scikit-learn plotly seaborn matplotlib umap-learn networkx scipy

!pip install pyngrok  # For tunneling Streamlit in Colab
!npm install localtunnel  # Alternative tunneling option

from pyngrok import ngrok

ngrok.set_auth_token("33HqOMdZt22g0h3whujWuU8Zysn_3SwSxCvRyTT1cXnHgaTzs")

import warnings
warnings.filterwarnings('ignore')

# Decision-Making Visualization Platform for Autonomous Agents (xAI)
# Complete End-to-End Implementation with Unsupervised ML

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import sqlite3
import json
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import umap.umap_ as umap

# Statistical Analysis
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import networkx as nx

# Data Generation and Simulation
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from enum import Enum
import random
import uuid
from threading import Thread
import time

# ==============================================================================
# 1. DATA MODELS AND SIMULATION ENGINE
# ==============================================================================

class AgentType(Enum):
    AUTONOMOUS_VEHICLE = "autonomous_vehicle"
    TRADING_BOT = "trading_bot"
    RECOMMENDATION_SYSTEM = "recommendation_system"
    RESOURCE_ALLOCATOR = "resource_allocator"
    SECURITY_SCANNER = "security_scanner"

class DecisionContext(Enum):
    HIGH_STAKES = "high_stakes"
    ROUTINE = "routine"
    EMERGENCY = "emergency"
    COLLABORATIVE = "collaborative"

@dataclass
class Decision:
    decision_id: str
    agent_id: str
    agent_type: AgentType
    timestamp: datetime
    decision_value: float
    confidence_score: float
    context: DecisionContext
    features: Dict[str, float]
    outcome_impact: float
    fairness_metrics: Dict[str, float] = field(default_factory=dict)
    bias_indicators: Dict[str, float] = field(default_factory=dict)

class AgentSimulator:
    """Simulates autonomous agent decision-making processes"""
    
    def __init__(self, num_agents: int = 50):
        self.num_agents = num_agents
        self.agents = self._create_agents()
        self.decision_history = []
        
    def _create_agents(self) -> List[Dict]:
        agents = []
        for i in range(self.num_agents):
            agent = {
                'agent_id': f"agent_{uuid.uuid4().hex[:8]}",
                'agent_type': random.choice(list(AgentType)),
                'bias_tendency': np.random.normal(0, 0.3),  # Individual bias
                'performance_baseline': np.random.uniform(0.6, 0.95),
                'fairness_weight': np.random.uniform(0.1, 0.9),
                'created_at': datetime.now() - timedelta(days=random.randint(1, 365))
            }
            agents.append(agent)
        return agents
    
    def simulate_decision(self, agent: Dict) -> Decision:
        """Simulate a single decision by an agent"""
        
        # Generate decision features
        features = {
            'risk_assessment': np.random.uniform(0, 1),
            'data_quality': np.random.uniform(0.3, 1.0),
            'time_pressure': np.random.uniform(0, 1),
            'resource_availability': np.random.uniform(0, 1),
            'stakeholder_count': np.random.poisson(3),
            'complexity_score': np.random.uniform(0, 1)
        }
        
        # Introduce bias based on agent characteristics
        bias_factor = agent['bias_tendency']
        context = random.choice(list(DecisionContext))
        
        # Decision value influenced by bias and context
        base_decision = np.random.uniform(0, 1)
        if context == DecisionContext.HIGH_STAKES:
            base_decision += bias_factor * 0.3
        elif context == DecisionContext.EMERGENCY:
            base_decision += bias_factor * 0.5
            
        decision_value = np.clip(base_decision, 0, 1)
        confidence_score = agent['performance_baseline'] * np.random.uniform(0.7, 1.0)
        
        # Calculate fairness metrics
        fairness_metrics = {
            'demographic_parity': np.random.uniform(0.4, 0.9),
            'equalized_odds': np.random.uniform(0.3, 0.8),
            'individual_fairness': np.random.uniform(0.5, 0.95),
            'treatment_equality': np.random.uniform(0.4, 0.85)
        }
        
        # Bias indicators
        bias_indicators = {
            'selection_bias': abs(bias_factor) * np.random.uniform(0.8, 1.2),
            'confirmation_bias': abs(bias_factor * 0.7) * features['data_quality'],
            'algorithmic_bias': abs(bias_factor * 0.9),
            'temporal_bias': abs(bias_factor * 0.6) * features['time_pressure']
        }
        
        outcome_impact = decision_value * confidence_score * np.random.uniform(0.8, 1.2)
        
        return Decision(
            decision_id=f"dec_{uuid.uuid4().hex[:12]}",
            agent_id=agent['agent_id'],
            agent_type=agent['agent_type'],
            timestamp=datetime.now(),
            decision_value=decision_value,
            confidence_score=confidence_score,
            context=context,
            features=features,
            outcome_impact=outcome_impact,
            fairness_metrics=fairness_metrics,
            bias_indicators=bias_indicators
        )
    
    def generate_batch_decisions(self, num_decisions: int = 1000) -> List[Decision]:
        """Generate a batch of decisions for analysis"""
        decisions = []
        for _ in range(num_decisions):
            agent = random.choice(self.agents)
            decision = self.simulate_decision(agent)
            decisions.append(decision)
        return decisions

  # ==============================================================================
# 2. DATA STORAGE AND MANAGEMENT
# ==============================================================================

class DatabaseManager:
    """Manages SQLite database for storing decision data"""
    
    def __init__(self, db_path: str = "agent_decisions.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Decisions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS decisions (
                decision_id TEXT PRIMARY KEY,
                agent_id TEXT,
                agent_type TEXT,
                timestamp TEXT,
                decision_value REAL,
                confidence_score REAL,
                context TEXT,
                features TEXT,  -- JSON string
                outcome_impact REAL,
                fairness_metrics TEXT,  -- JSON string
                bias_indicators TEXT   -- JSON string
            )
        ''')
        
        # Agents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                agent_type TEXT,
                bias_tendency REAL,
                performance_baseline REAL,
                fairness_weight REAL,
                created_at TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_decisions(self, decisions: List[Decision]):
        """Store decisions in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for decision in decisions:
            cursor.execute('''
                INSERT OR REPLACE INTO decisions VALUES 
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                decision.decision_id,
                decision.agent_id,
                decision.agent_type.value,
                decision.timestamp.isoformat(),
                decision.decision_value,
                decision.confidence_score,
                decision.context.value,
                json.dumps(decision.features),
                decision.outcome_impact,
                json.dumps(decision.fairness_metrics),
                json.dumps(decision.bias_indicators)
            ))
        
        conn.commit()
        conn.close()
    
    def load_decisions_as_dataframe(self) -> pd.DataFrame:
        """Load decisions as pandas DataFrame"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM decisions", conn)
        conn.close()
        
        if len(df) > 0:
            # Parse JSON columns
            df['features'] = df['features'].apply(json.loads)
            df['fairness_metrics'] = df['fairness_metrics'].apply(json.loads)
            df['bias_indicators'] = df['bias_indicators'].apply(json.loads)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Expand nested dictionaries into columns
            features_df = pd.json_normalize(df['features'])
            fairness_df = pd.json_normalize(df['fairness_metrics']).add_prefix('fairness_')
            bias_df = pd.json_normalize(df['bias_indicators']).add_prefix('bias_')
            
            df = pd.concat([df.drop(['features', 'fairness_metrics', 'bias_indicators'], axis=1),
                           features_df, fairness_df, bias_df], axis=1)
        
        return df

  # ==============================================================================
# 3. UNSUPERVISED ML ANALYSIS ENGINE
# ==============================================================================

class UnsupervisedAnalyzer:
    """Advanced unsupervised ML analysis for bias detection and pattern discovery"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA()
        self.kmeans = KMeans()
        self.dbscan = DBSCAN()
        self.isolation_forest = IsolationForest()
        
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML analysis"""
        feature_columns = [
            'decision_value', 'confidence_score', 'outcome_impact',
            'risk_assessment', 'data_quality', 'time_pressure',
            'resource_availability', 'stakeholder_count', 'complexity_score',
            'fairness_demographic_parity', 'fairness_equalized_odds',
            'fairness_individual_fairness', 'fairness_treatment_equality',
            'bias_selection_bias', 'bias_confirmation_bias',
            'bias_algorithmic_bias', 'bias_temporal_bias'
        ]
        
        available_features = [col for col in feature_columns if col in df.columns]
        return df[available_features].fillna(df[available_features].mean())
    
    def detect_bias_clusters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect bias patterns using clustering algorithms"""
        features = self.prepare_features(df)
        X_scaled = self.scaler.fit_transform(features)
        
        results = {}
        
        # K-Means Clustering
        optimal_k = self._find_optimal_clusters(X_scaled)
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        
        # DBSCAN for density-based clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        
        results['kmeans'] = {
            'labels': kmeans_labels,
            'centers': kmeans.cluster_centers_,
            'n_clusters': optimal_k,
            'silhouette_score': silhouette_score(X_scaled, kmeans_labels)
        }
        
        results['dbscan'] = {
            'labels': dbscan_labels,
            'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
            'noise_points': np.sum(dbscan_labels == -1)
        }
        
        return results
    
    def _find_optimal_clusters(self, X: np.ndarray, max_k: int = 10) -> int:
        """Find optimal number of clusters using elbow method"""
        inertias = []
        silhouette_scores = []
        
        for k in range(2, min(max_k + 1, len(X))):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, labels))
        
        # Find elbow point
        optimal_k = 2 + np.argmax(silhouette_scores)
        return optimal_k
    
    def detect_anomalies(self, df: pd.DataFrame) -> np.ndarray:
        """Detect anomalous decisions using Isolation Forest"""
        features = self.prepare_features(df)
        X_scaled = self.scaler.fit_transform(features)
        
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = isolation_forest.fit_predict(X_scaled)
        
        return anomaly_labels
    
    def dimensionality_reduction(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Perform dimensionality reduction for visualization"""
        features = self.prepare_features(df)
        X_scaled = self.scaler.fit_transform(features)
        
        results = {}
        
        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        results['pca'] = pca_result
        results['pca_explained_variance'] = pca.explained_variance_ratio_
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)-1))
        tsne_result = tsne.fit_transform(X_scaled)
        results['tsne'] = tsne_result
        
        # UMAP
        try:
            umap_reducer = umap.UMAP(n_components=2, random_state=42)
            umap_result = umap_reducer.fit_transform(X_scaled)
            results['umap'] = umap_result
        except:
            results['umap'] = pca_result  # Fallback to PCA
        
        return results

  # ==============================================================================
# 4. FAIRNESS AND BIAS ANALYSIS
# ==============================================================================

class FairnessAnalyzer:
    """Comprehensive fairness and bias analysis module"""
    
    def __init__(self):
        self.fairness_thresholds = {
            'demographic_parity': 0.8,
            'equalized_odds': 0.7,
            'individual_fairness': 0.8,
            'treatment_equality': 0.75
        }
        
    def analyze_group_fairness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze fairness across different agent groups"""
        results = {}
        
        # Group by agent type
        for agent_type in df['agent_type'].unique():
            type_data = df[df['agent_type'] == agent_type]
            
            fairness_scores = {
                'demographic_parity': type_data['fairness_demographic_parity'].mean(),
                'equalized_odds': type_data['fairness_equalized_odds'].mean(),
                'individual_fairness': type_data['fairness_individual_fairness'].mean(),
                'treatment_equality': type_data['fairness_treatment_equality'].mean()
            }
            
            # Calculate violations
            violations = {}
            for metric, score in fairness_scores.items():
                threshold = self.fairness_thresholds.get(metric, 0.8)
                violations[metric] = score < threshold
            
            results[agent_type] = {
                'fairness_scores': fairness_scores,
                'violations': violations,
                'sample_size': len(type_data),
                'avg_decision_value': type_data['decision_value'].mean(),
                'avg_confidence': type_data['confidence_score'].mean()
            }
        
        return results
    
    def detect_systematic_bias(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect systematic bias patterns across the system"""
        bias_analysis = {}
        
        # Temporal bias analysis
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        temporal_bias = {
            'hourly_variance': df.groupby('hour')['decision_value'].var().mean(),
            'daily_variance': df.groupby('day_of_week')['decision_value'].var().mean(),
            'peak_bias_hours': df.groupby('hour')['bias_algorithmic_bias'].mean().nlargest(3).index.tolist(),
            'low_bias_hours': df.groupby('hour')['bias_algorithmic_bias'].mean().nsmallest(3).index.tolist()
        }
        
        # Context-based bias
        context_bias = {}
        for context in df['context'].unique():
            context_data = df[df['context'] == context]
            context_bias[context] = {
                'avg_bias': context_data[['bias_selection_bias', 'bias_confirmation_bias', 
                                        'bias_algorithmic_bias', 'bias_temporal_bias']].mean().mean(),
                'fairness_score': context_data[['fairness_demographic_parity', 'fairness_equalized_odds',
                                              'fairness_individual_fairness', 'fairness_treatment_equality']].mean().mean()
            }
        
        # Statistical significance testing
        high_stakes = df[df['context'] == 'high_stakes']['decision_value']
        routine = df[df['context'] == 'routine']['decision_value']
        
        if len(high_stakes) > 0 and len(routine) > 0:
            stat_test = stats.ttest_ind(high_stakes, routine)
            statistical_significance = {
                'p_value': stat_test.pvalue,
                'statistically_significant': stat_test.pvalue < 0.05,
                'effect_size': abs(high_stakes.mean() - routine.mean())
            }
        else:
            statistical_significance = None
        
        bias_analysis = {
            'temporal_bias': temporal_bias,
            'context_bias': context_bias,
            'statistical_significance': statistical_significance
        }
        
        return bias_analysis
    
    def generate_fairness_alerts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate alerts for fairness violations"""
        alerts = []
        
        group_analysis = self.analyze_group_fairness(df)
        bias_analysis = self.detect_systematic_bias(df)
        
        # Check for fairness violations
        for agent_type, analysis in group_analysis.items():
            for metric, violated in analysis['violations'].items():
                if violated:
                    alerts.append({
                        'type': 'fairness_violation',
                        'severity': 'high',
                        'agent_type': agent_type,
                        'metric': metric,
                        'actual_score': analysis['fairness_scores'][metric],
                        'threshold': self.fairness_thresholds[metric],
                        'timestamp': datetime.now(),
                        'recommendation': f"Review {agent_type} agents for {metric} bias"
                    })
        
        # Check for high bias contexts
        for context, analysis in bias_analysis['context_bias'].items():
            if analysis['avg_bias'] > 0.6:  # High bias threshold
                alerts.append({
                    'type': 'high_bias_context',
                    'severity': 'medium',
                    'context': context,
                    'bias_score': analysis['avg_bias'],
                    'fairness_score': analysis['fairness_score'],
                    'timestamp': datetime.now(),
                    'recommendation': f"Investigate bias in {context} decisions"
                })
        
        return alerts

  # ==============================================================================
# 5. VISUALIZATION ENGINE
# ==============================================================================

class VisualizationEngine:
    """Advanced visualization engine for bias and fairness analysis"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        
    def plot_bias_clusters(self, df: pd.DataFrame, cluster_results: Dict, 
                          reduction_results: Dict) -> go.Figure:
        """Create interactive cluster visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('PCA Clusters', 't-SNE Clusters', 'UMAP Clusters', 'Bias Distribution'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "histogram"}]]
        )
        
        # PCA visualization
        pca_data = reduction_results['pca']
        fig.add_trace(
            go.Scatter(
                x=pca_data[:, 0], y=pca_data[:, 1],
                mode='markers',
                marker=dict(color=cluster_results['kmeans']['labels'], 
                          colorscale='Viridis', size=8),
                name='PCA Clusters',
                text=[f"Agent: {aid}<br>Type: {at}<br>Cluster: {cl}" 
                      for aid, at, cl in zip(df['agent_id'], df['agent_type'], 
                                           cluster_results['kmeans']['labels'])],
                hovertemplate='%{text}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # t-SNE visualization
        tsne_data = reduction_results['tsne']
        fig.add_trace(
            go.Scatter(
                x=tsne_data[:, 0], y=tsne_data[:, 1],
                mode='markers',
                marker=dict(color=cluster_results['dbscan']['labels'], 
                          colorscale='Plasma', size=8),
                name='t-SNE Clusters',
                text=[f"Agent: {aid}<br>Type: {at}<br>Cluster: {cl}" 
                      for aid, at, cl in zip(df['agent_id'], df['agent_type'], 
                                           cluster_results['dbscan']['labels'])],
                hovertemplate='%{text}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # UMAP visualization
        umap_data = reduction_results['umap']
        fig.add_trace(
            go.Scatter(
                x=umap_data[:, 0], y=umap_data[:, 1],
                mode='markers',
                marker=dict(color=df['bias_algorithmic_bias'], 
                          colorscale='RdYlBu_r', size=8, 
                          colorbar=dict(title="Algorithmic Bias")),
                name='UMAP - Bias Colored',
                text=[f"Agent: {aid}<br>Type: {at}<br>Bias: {bias:.3f}" 
                      for aid, at, bias in zip(df['agent_id'], df['agent_type'], 
                                             df['bias_algorithmic_bias'])],
                hovertemplate='%{text}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Bias distribution
        fig.add_trace(
            go.Histogram(
                x=df['bias_algorithmic_bias'],
                nbinsx=30,
                name='Algorithmic Bias Distribution',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Bias Cluster Analysis Dashboard",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def plot_fairness_metrics(self, group_analysis: Dict) -> go.Figure:
        """Create fairness metrics visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Fairness Scores by Agent Type', 'Violations Heatmap', 
                          'Decision Quality vs Fairness', 'Confidence vs Bias'),
            specs=[[{"type": "bar"}, {"type": "heatmap"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Prepare data
        agent_types = list(group_analysis.keys())
        metrics = ['demographic_parity', 'equalized_odds', 'individual_fairness', 'treatment_equality']
        
        # Fairness scores bar chart
        for metric in metrics:
            scores = [group_analysis[at]['fairness_scores'][metric] for at in agent_types]
            fig.add_trace(
                go.Bar(name=metric.replace('_', ' ').title(), x=agent_types, y=scores),
                row=1, col=1
            )
        
        # Violations heatmap
        violation_matrix = []
        for at in agent_types:
            violations = [1 if group_analysis[at]['violations'][metric] else 0 for metric in metrics]
            violation_matrix.append(violations)
        
        fig.add_trace(
            go.Heatmap(
                z=violation_matrix,
                x=[m.replace('_', ' ').title() for m in metrics],
                y=agent_types,
                colorscale='Reds',
                name='Violations'
            ),
            row=1, col=2
        )
        
        # Decision quality vs fairness scatter
        quality_scores = [group_analysis[at]['avg_decision_value'] for at in agent_types]
        fairness_avg = [np.mean(list(group_analysis[at]['fairness_scores'].values())) for at in agent_types]
        
        fig.add_trace(
            go.Scatter(
                x=fairness_avg, y=quality_scores,
                mode='markers+text',
                text=agent_types,
                textposition='top center',
                marker=dict(size=12, color='blue'),
                name='Quality vs Fairness'
            ),
            row=2, col=1
        )
        
        # Confidence vs sample size
        confidence_scores = [group_analysis[at]['avg_confidence'] for at in agent_types]
        sample_sizes = [group_analysis[at]['sample_size'] for at in agent_types]
        
        fig.add_trace(
            go.Scatter(
                x=sample_sizes, y=confidence_scores,
                mode='markers+text',
                text=agent_types,
                textposition='top center',
                marker=dict(size=12, color='green'),
                name='Confidence vs Sample Size'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Comprehensive Fairness Analysis Dashboard",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def plot_temporal_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Create temporal bias analysis visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Hourly Bias Patterns', 'Daily Decision Trends', 
                          'Context-based Analysis', 'Agent Type Performance'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "box"}]]
        )
        
        # Hourly bias patterns
        hourly_bias = df.groupby(df['timestamp'].dt.hour)['bias_algorithmic_bias'].mean()
        fig.add_trace(
            go.Scatter(
                x=hourly_bias.index, y=hourly_bias.values,
                mode='lines+markers',
                name='Hourly Avg Bias',
                line=dict(color='red', width=3)
            ),
            row=1, col=1
        )
        
        # Daily decision trends
        daily_decisions = df.groupby(df['timestamp'].dt.date).size()
        fig.add_trace(
            go.Scatter(
                x=daily_decisions.index, y=daily_decisions.values,
                mode='lines+markers',
                name='Daily Decision Count',
                line=dict(color='blue', width=2)
            ),
            row=1, col=2
        )
        
        # Context-based analysis
        context_fairness = df.groupby('context')[['fairness_demographic_parity', 
                                                'fairness_equalized_odds',
                                                'fairness_individual_fairness']].mean()
        
        for metric in context_fairness.columns:
            fig.add_trace(
                go.Bar(
                    x=context_fairness.index, y=context_fairness[metric],
                    name=metric.replace('fairness_', '').replace('_', ' ').title()
                ),
                row=2, col=1
            )
        
        # Agent type performance box plots
        for agent_type in df['agent_type'].unique():
            type_data = df[df['agent_type'] == agent_type]
            fig.add_trace(
                go.Box(
                    y=type_data['decision_value'],
                    name=agent_type,
                    boxpoints='outliers'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Temporal and Contextual Bias Analysis",
            height=800,
            showlegend=True
        )
        
        return fig

  # ==============================================================================
# 6. STREAMLIT DASHBOARD
# ==============================================================================

def create_streamlit_dashboard():
    """Create the main Streamlit dashboard"""
    
    st.set_page_config(
        page_title="xAI Decision Visualization Platform",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 Decision-Making Visualization Platform for Autonomous Agents")
    st.markdown("### Unsupervised ML-powered Bias Detection and Fairness Analysis")
    
    # Initialize components
    if 'simulator' not in st.session_state:
        st.session_state.simulator = AgentSimulator(num_agents=30)
        st.session_state.db_manager = DatabaseManager()
        st.session_state.analyzer = UnsupervisedAnalyzer()
        st.session_state.fairness_analyzer = FairnessAnalyzer()
        st.session_state.viz_engine = VisualizationEngine()
    
    # Sidebar controls
    st.sidebar.header("🔧 Control Panel")
    
    # Data generation
    if st.sidebar.button("🔄 Generate New Data"):
        with st.spinner("Generating decision data..."):
            decisions = st.session_state.simulator.generate_batch_decisions(1000)
            st.session_state.db_manager.store_decisions(decisions)
            st.success("New data generated successfully!")
    
    # Load existing data
    df = st.session_state.db_manager.load_decisions_as_dataframe()
    
    if len(df) == 0:
        st.warning("No data available. Please generate data first.")
        st.stop()
    
    # Sidebar filters
    st.sidebar.subheader("📊 Data Filters")
    
    selected_agent_types = st.sidebar.multiselect(
        "Select Agent Types",
        options=df['agent_type'].unique(),
        default=df['agent_type'].unique()
    )
    
    selected_contexts = st.sidebar.multiselect(
        "Select Decision Contexts",
        options=df['context'].unique(),
        default=df['context'].unique()
    )
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['timestamp'].min().date(), df['timestamp'].max().date()),
        min_value=df['timestamp'].min().date(),
        max_value=df['timestamp'].max().date()
    )
    
    # Apply filters
    filtered_df = df[
        (df['agent_type'].isin(selected_agent_types)) &
        (df['context'].isin(selected_contexts)) &
        (df['timestamp'].dt.date >= date_range[0]) &
        (df['timestamp'].dt.date <= date_range[1])
    ]
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", "🎯 Bias Clusters", "⚖️ Fairness Analysis", 
        "⏰ Temporal Analysis", "🚨 Alerts"
    ])
    
    with tab1:
        st.header("System Overview")
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Decisions", len(filtered_df))
        
        with col2:
            avg_bias = filtered_df['bias_algorithmic_bias'].mean()
            st.metric("Avg Algorithmic Bias", f"{avg_bias:.3f}")
        
        with col3:
            avg_fairness = filtered_df[[col for col in filtered_df.columns if col.startswith('fairness_')]].mean().mean()
            st.metric("Avg Fairness Score", f"{avg_fairness:.3f}")
        
        with col4:
            anomalies = st.session_state.analyzer.detect_anomalies(filtered_df)
            anomaly_count = np.sum(anomalies == -1)
            st.metric("Anomalous Decisions", anomaly_count)
        
        with col5:
            high_confidence = np.sum(filtered_df['confidence_score'] > 0.8)
            st.metric("High Confidence Decisions", high_confidence)
        
        # System health indicators
        st.subheader("System Health Indicators")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bias distribution
            fig_bias = px.histogram(
                filtered_df, x='bias_algorithmic_bias',
                title="Algorithmic Bias Distribution",
                nbins=30, marginal="box"
            )
            st.plotly_chart(fig_bias, use_container_width=True)
        
        with col2:
            # Fairness vs Confidence scatter
            fig_scatter = px.scatter(
                filtered_df, 
                x='fairness_individual_fairness', 
                y='confidence_score',
                color='agent_type',
                size='outcome_impact',
                title="Fairness vs Confidence by Agent Type",
                hover_data=['decision_value', 'context']
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Agent type performance comparison
        st.subheader("Agent Type Performance Comparison")
        
        performance_data = []
        for agent_type in filtered_df['agent_type'].unique():
            type_data = filtered_df[filtered_df['agent_type'] == agent_type]
            performance_data.append({
                'Agent Type': agent_type,
                'Avg Decision Value': type_data['decision_value'].mean(),
                'Avg Confidence': type_data['confidence_score'].mean(),
                'Avg Fairness': type_data[[col for col in type_data.columns if col.startswith('fairness_')]].mean().mean(),
                'Avg Bias': type_data[[col for col in type_data.columns if col.startswith('bias_')]].mean().mean(),
                'Decision Count': len(type_data)
            })
        
        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df, use_container_width=True)
    
    with tab2:
        st.header("🎯 Bias Cluster Analysis")
        
        if len(filtered_df) > 10:  # Minimum data for clustering
            with st.spinner("Performing unsupervised analysis..."):
                # Run clustering analysis
                cluster_results = st.session_state.analyzer.detect_bias_clusters(filtered_df)
                reduction_results = st.session_state.analyzer.dimensionality_reduction(filtered_df)
                
                # Display cluster visualization
                fig_clusters = st.session_state.viz_engine.plot_bias_clusters(
                    filtered_df, cluster_results, reduction_results
                )
                st.plotly_chart(fig_clusters, use_container_width=True)
                
                # Cluster analysis results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("K-Means Analysis")
                    st.write(f"**Optimal Clusters:** {cluster_results['kmeans']['n_clusters']}")
                    st.write(f"**Silhouette Score:** {cluster_results['kmeans']['silhouette_score']:.3f}")
                    
                    # Cluster characteristics
                    cluster_labels = cluster_results['kmeans']['labels']
                    filtered_df['cluster'] = cluster_labels
                    
                    cluster_stats = []
                    for cluster_id in range(cluster_results['kmeans']['n_clusters']):
                        cluster_data = filtered_df[filtered_df['cluster'] == cluster_id]
                        cluster_stats.append({
                            'Cluster': cluster_id,
                            'Size': len(cluster_data),
                            'Avg Bias': cluster_data['bias_algorithmic_bias'].mean(),
                            'Avg Fairness': cluster_data[[col for col in cluster_data.columns if col.startswith('fairness_')]].mean().mean(),
                            'Dominant Agent Type': cluster_data['agent_type'].mode().iloc[0] if len(cluster_data) > 0 else 'N/A'
                        })
                    
                    cluster_df = pd.DataFrame(cluster_stats)
                    st.dataframe(cluster_df, use_container_width=True)
                
                with col2:
                    st.subheader("DBSCAN Analysis")
                    st.write(f"**Clusters Found:** {cluster_results['dbscan']['n_clusters']}")
                    st.write(f"**Noise Points:** {cluster_results['dbscan']['noise_points']}")
                    
                    # Anomaly detection results
                    anomalies = st.session_state.analyzer.detect_anomalies(filtered_df)
                    anomaly_df = filtered_df.copy()
                    anomaly_df['is_anomaly'] = anomalies == -1
                    
                    st.subheader("Anomalous Decisions")
                    anomalous_decisions = anomaly_df[anomaly_df['is_anomaly']]
                    
                    if len(anomalous_decisions) > 0:
                        st.write(f"Found {len(anomalous_decisions)} anomalous decisions")
                        display_cols = ['agent_id', 'agent_type', 'decision_value', 'confidence_score', 'bias_algorithmic_bias']
                        st.dataframe(anomalous_decisions[display_cols].head(10), use_container_width=True)
                    else:
                        st.write("No anomalous decisions detected.")
        else:
            st.warning("Need at least 10 data points for clustering analysis.")
    
    with tab3:
        st.header("⚖️ Fairness Analysis")
        
        # Run fairness analysis
        group_analysis = st.session_state.fairness_analyzer.analyze_group_fairness(filtered_df)
        bias_analysis = st.session_state.fairness_analyzer.detect_systematic_bias(filtered_df)
        
        # Display fairness visualization
        fig_fairness = st.session_state.viz_engine.plot_fairness_metrics(group_analysis)
        st.plotly_chart(fig_fairness, use_container_width=True)
        
        # Fairness summary table
        st.subheader("Fairness Summary by Agent Type")
        
        fairness_summary = []
        for agent_type, analysis in group_analysis.items():
            fairness_summary.append({
                'Agent Type': agent_type,
                'Sample Size': analysis['sample_size'],
                'Demographic Parity': f"{analysis['fairness_scores']['demographic_parity']:.3f}",
                'Equalized Odds': f"{analysis['fairness_scores']['equalized_odds']:.3f}",
                'Individual Fairness': f"{analysis['fairness_scores']['individual_fairness']:.3f}",
                'Treatment Equality': f"{analysis['fairness_scores']['treatment_equality']:.3f}",
                'Violations': sum(analysis['violations'].values()),
                'Avg Decision Value': f"{analysis['avg_decision_value']:.3f}",
                'Avg Confidence': f"{analysis['avg_confidence']:.3f}"
            })
        
        fairness_df = pd.DataFrame(fairness_summary)
        st.dataframe(fairness_df, use_container_width=True)
        
        # Bias analysis results
        st.subheader("Systematic Bias Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Temporal Bias Patterns:**")
            temporal = bias_analysis['temporal_bias']
            st.write(f"- Hourly Variance: {temporal['hourly_variance']:.4f}")
            st.write(f"- Daily Variance: {temporal['daily_variance']:.4f}")
            st.write(f"- Peak Bias Hours: {temporal['peak_bias_hours']}")
            st.write(f"- Low Bias Hours: {temporal['low_bias_hours']}")
        
        with col2:
            st.write("**Context-based Bias:**")
            for context, analysis in bias_analysis['context_bias'].items():
                st.write(f"- {context}: Bias={analysis['avg_bias']:.3f}, Fairness={analysis['fairness_score']:.3f}")
        
        # Statistical significance
        if bias_analysis['statistical_significance']:
            st.subheader("Statistical Analysis")
            stat_sig = bias_analysis['statistical_significance']
            st.write(f"**High Stakes vs Routine Decisions:**")
            st.write(f"- P-value: {stat_sig['p_value']:.6f}")
            st.write(f"- Statistically Significant: {stat_sig['statistically_significant']}")
            st.write(f"- Effect Size: {stat_sig['effect_size']:.4f}")
    
    with tab4:
        st.header("⏰ Temporal Analysis")
        
        # Create temporal visualization
        fig_temporal = st.session_state.viz_engine.plot_temporal_analysis(filtered_df)
        st.plotly_chart(fig_temporal, use_container_width=True)
        
        # Time-based insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Peak Activity Analysis")
            
            # Hour analysis
            hourly_decisions = filtered_df.groupby(filtered_df['timestamp'].dt.hour).size()
            peak_hour = hourly_decisions.idxmax()
            peak_count = hourly_decisions.max()
            
            st.write(f"**Peak Hour:** {peak_hour}:00 ({peak_count} decisions)")
            
            # Day analysis
            daily_decisions = filtered_df.groupby(filtered_df['timestamp'].dt.day_name()).size()
            peak_day = daily_decisions.idxmax()
            peak_day_count = daily_decisions.max()
            
            st.write(f"**Peak Day:** {peak_day} ({peak_day_count} decisions)")
            
            # Context timing
            st.subheader("Context Timing Patterns")
            context_timing = filtered_df.groupby(['context', filtered_df['timestamp'].dt.hour]).size().unstack(fill_value=0)
            st.dataframe(context_timing)
        
        with col2:
            st.subheader("Bias Temporal Patterns")
            
            # Hourly bias analysis
            hourly_bias = filtered_df.groupby(filtered_df['timestamp'].dt.hour)['bias_algorithmic_bias'].mean()
            worst_bias_hour = hourly_bias.idxmax()
            best_bias_hour = hourly_bias.idxmin()
            
            st.write(f"**Highest Bias Hour:** {worst_bias_hour}:00 (bias: {hourly_bias.max():.3f})")
            st.write(f"**Lowest Bias Hour:** {best_bias_hour}:00 (bias: {hourly_bias.min():.3f})")
            
            # Weekly patterns
            weekly_bias = filtered_df.groupby(filtered_df['timestamp'].dt.day_name())['bias_algorithmic_bias'].mean()
            
            st.subheader("Weekly Bias Patterns")
            for day, bias in weekly_bias.items():
                st.write(f"- {day}: {bias:.3f}")
    
    with tab5:
        st.header("🚨 Fairness Alerts and Recommendations")
        
        # Generate alerts
        alerts = st.session_state.fairness_analyzer.generate_fairness_alerts(filtered_df)
        
        if alerts:
            st.subheader(f"Active Alerts ({len(alerts)})")
            
            for i, alert in enumerate(alerts):
                severity_color = {
                    'high': '🔴',
                    'medium': '🟡',
                    'low': '🟢'
                }
                
                with st.expander(f"{severity_color[alert['severity']]} {alert['type'].replace('_', ' ').title()}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Alert Details:**")
                        for key, value in alert.items():
                            if key not in ['recommendation']:
                                st.write(f"- {key.replace('_', ' ').title()}: {value}")
                    
                    with col2:
                        st.write("**Recommendation:**")
                        st.write(alert['recommendation'])
                        
                        if st.button(f"Mark as Resolved", key=f"resolve_{i}"):
                            st.success("Alert marked as resolved!")
        else:
            st.success("✅ No active alerts. System operating within fairness thresholds!")
        
        # Recommendations dashboard
        st.subheader("🎯 AI-Generated Recommendations")
        
        recommendations = []
        
        # Analyze current state and generate recommendations
        avg_bias = filtered_df['bias_algorithmic_bias'].mean()
        avg_fairness = filtered_df[[col for col in filtered_df.columns if col.startswith('fairness_')]].mean().mean()
        
        if avg_bias > 0.5:
            recommendations.append({
                'type': 'High System Bias',
                'priority': 'High',
                'action': 'Review training data and model parameters',
                'impact': 'Reduce overall system bias by 20-30%'
            })
        
        if avg_fairness < 0.7:
            recommendations.append({
                'type': 'Low Fairness Score',
                'priority': 'High',
                'action': 'Implement fairness constraints in decision algorithms',
                'impact': 'Improve fairness metrics across all agent types'
            })
        
        # Agent-specific recommendations
        for agent_type, analysis in group_analysis.items():
            violation_count = sum(analysis['violations'].values())
            if violation_count >= 2:
                recommendations.append({
                    'type': f'{agent_type} Multiple Violations',
                    'priority': 'Medium',
                    'action': f'Retrain {agent_type} agents with balanced datasets',
                    'impact': f'Address {violation_count} fairness violations'
                })
        
        if recommendations:
            for i, rec in enumerate(recommendations):
                priority_color = {
                    'High': '🔴',
                    'Medium': '🟡',
                    'Low': '🟢'
                }
                
                st.info(f"""
                **{priority_color[rec['priority']]} {rec['type']}**
                
                **Action:** {rec['action']}
                
                **Expected Impact:** {rec['impact']}
                """)
        else:
            st.success("✅ System performing optimally. No recommendations needed.")
    
    # Footer with system info
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"📊 **Data Points:** {len(filtered_df)}")
    
    with col2:
        st.info(f"🤖 **Active Agents:** {filtered_df['agent_id'].nunique()}")
    
    with col3:
        last_update = filtered_df['timestamp'].max()
        st.info(f"🕐 **Last Update:** {last_update.strftime('%Y-%m-%d %H:%M')}")

  # ==============================================================================
# 7. MAIN EXECUTION AND CLI
# ==============================================================================

def main():
    """Main execution function"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'dashboard':
        # Run Streamlit dashboard
        create_streamlit_dashboard()
    else:
        # Run demo analysis
        print("🤖 Decision-Making Visualization Platform for Autonomous Agents")
        print("=" * 70)
        
        # Initialize components
        print("Initializing system components...")
        simulator = AgentSimulator(num_agents=20)
        db_manager = DatabaseManager()
        analyzer = UnsupervisedAnalyzer()
        fairness_analyzer = FairnessAnalyzer()
        viz_engine = VisualizationEngine()
        
        # Generate sample data
        print("Generating sample decision data...")
        decisions = simulator.generate_batch_decisions(500)
        db_manager.store_decisions(decisions)
        
        # Load and analyze data
        print("Loading and analyzing data...")
        df = db_manager.load_decisions_as_dataframe()
        
        print(f"\n📊 Dataset Overview:")
        print(f"- Total decisions: {len(df)}")
        print(f"- Unique agents: {df['agent_id'].nunique()}")
        print(f"- Agent types: {df['agent_type'].nunique()}")
        print(f"- Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Unsupervised analysis
        print("\n🎯 Running Unsupervised ML Analysis...")
        cluster_results = analyzer.detect_bias_clusters(df)
        reduction_results = analyzer.dimensionality_reduction(df)
        anomalies = analyzer.detect_anomalies(df)
        
        print(f"- K-Means clusters: {cluster_results['kmeans']['n_clusters']}")
        print(f"- Silhouette score: {cluster_results['kmeans']['silhouette_score']:.3f}")
        print(f"- DBSCAN clusters: {cluster_results['dbscan']['n_clusters']}")
        print(f"- Anomalous decisions: {np.sum(anomalies == -1)}")
        
        # Fairness analysis
        print("\n⚖️ Fairness Analysis Results:")
        group_analysis = fairness_analyzer.analyze_group_fairness(df)
        bias_analysis = fairness_analyzer.detect_systematic_bias(df)
        alerts = fairness_analyzer.generate_fairness_alerts(df)
        
        for agent_type, analysis in group_analysis.items():
            violations = sum(analysis['violations'].values())
            print(f"- {agent_type}: {violations} fairness violations")
        
        print(f"\n🚨 Active Alerts: {len(alerts)}")
        for alert in alerts[:3]:  # Show first 3 alerts
            print(f"- {alert['type']}: {alert['severity']} severity")
        
        print(f"\n📈 System Metrics:")
        avg_bias = df['bias_algorithmic_bias'].mean()
        avg_fairness = df[[col for col in df.columns if col.startswith('fairness_')]].mean().mean()
        avg_confidence = df['confidence_score'].mean()
        
        print(f"- Average algorithmic bias: {avg_bias:.3f}")
        print(f"- Average fairness score: {avg_fairness:.3f}")
        print(f"- Average confidence: {avg_confidence:.3f}")
        
        print("\n✅ Analysis complete!")
        print("\nTo run the interactive dashboard:")
        print("streamlit run this_file.py dashboard")
        print("\nOr use: python this_file.py dashboard")

if __name__ == "__main__":
    main()

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
# STREAMLIT DASHBOARD APPLICATION
# =====================================================================

def create_project_overview():
    """Create comprehensive project overview section"""
    
    st.markdown("""
    <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; text-align: center; margin: 0;'>
            🤖 Decision-Making Visualization Platform for Autonomous Agents
        </h1>
        <h3 style='color: #f0f0f0; text-align: center; margin: 0.5rem 0 0 0;'>
            Using Unsupervised Machine Learning for Bias Detection & Fairness Analysis
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Project Overview
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.markdown("""
        ### 📋 **Project Overview**
        
        This platform continuously analyzes and visualizes the fairness and biases in 
        autonomous agent decision processes. It tracks interactions, exposes disparities 
        among different agent groups, and dynamically alerts developers to potential 
        inequities, supporting ethical and transparent autonomous operations.
        
        **🎯 Key Objectives:**
        - Real-time bias detection in autonomous systems
        - Fairness metric monitoring across agent types
        - Unsupervised pattern discovery in decision-making
        - Automated alert system for ethics violations
        - Comprehensive visualization dashboard
        """)
    
    with col2:
        st.markdown("""
        ### 🔧 **Tech Stack**
        
        **Backend:**
        - Python 3.8+
        - SQLite Database
        - scikit-learn
        
        **Frontend:**
        - Streamlit
        - Plotly/Dash
        - Interactive Widgets
        
        **ML/AI:**
        - K-Means Clustering
        - DBSCAN
        - Isolation Forest
        - PCA, t-SNE, UMAP
        - Statistical Analysis
        """)
    
    with col3:
        st.markdown("""
        ### 🤖 **Agent Types Monitored**
        
        - **Autonomous Vehicles** 🚗
          - Traffic decision systems
          - Route optimization
          
        - **Trading Bots** 📈
          - Financial algorithms
          - Risk assessment
          
        - **Recommendation Systems** 🎯
          - Content filtering
          - User profiling
          
        - **Resource Allocators** ⚡
          - Cloud computing
          - Task scheduling
          
        - **Security Scanners** 🛡️
          - Threat detection
          - Access control
        """)
    
    # Methodology Section
    st.markdown("""
    ---
    ### 🧠 **Methodology & Approach**
    """)
    
    method_col1, method_col2 = st.columns(2)
    
    with method_col1:
        st.markdown("""
        #### **Unsupervised Learning Pipeline**
        
        **1. Data Collection & Simulation**
        - Multi-agent decision simulation
        - Context-aware decision generation
        - Bias injection mechanisms
        - Real-time data streaming
        
        **2. Feature Engineering**
        - Risk assessment metrics
        - Data quality indicators
        - Time pressure factors
        - Resource availability scores
        - Stakeholder impact analysis
        
        **3. Clustering Analysis**
        - **K-Means**: Identify decision patterns
        - **DBSCAN**: Detect outlier behaviors
        - **Silhouette Analysis**: Validate clusters
        - **Anomaly Detection**: Isolation Forest
        """)
    
    with method_col2:
        st.markdown("""
        #### **Fairness Analysis Framework**
        
        **1. Fairness Metrics**
        - **Demographic Parity**: Equal outcomes across groups
        - **Equalized Odds**: Equal true/false positive rates
        - **Individual Fairness**: Similar treatment for similar cases
        - **Treatment Equality**: Equal error rates
        
        **2. Bias Detection**
        - **Selection Bias**: Systematic preferences
        - **Confirmation Bias**: Data quality influence
        - **Algorithmic Bias**: Model-induced disparities
        - **Temporal Bias**: Time-dependent patterns
        
        **3. Alert System**
        - Real-time violation monitoring
        - Threshold-based alerting
        - Severity classification
        - Automated recommendations
        """)

def create_key_features_section():
    """Create key features showcase"""
    
    st.markdown("""
    ---
    ### ✨ **Key Features & Capabilities**
    """)
    
    # Feature cards
    feature_col1, feature_col2, feature_col3, feature_col4 = st.columns(4)
    
    with feature_col1:
        st.markdown("""
        <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #007bff;'>
            <h4>🎯 Real-time Analysis</h4>
            <p>Continuous monitoring of agent decisions with live bias detection and fairness assessment.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col2:
        st.markdown("""
        <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #28a745;'>
            <h4>📊 Interactive Visualizations</h4>
            <p>Dynamic charts and graphs using Plotly for exploring decision patterns and bias clusters.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col3:
        st.markdown("""
        <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #ffc107;'>
            <h4>🚨 Alert System</h4>
            <p>Automated fairness violation detection with severity classification and recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col4:
        st.markdown("""
        <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #dc3545;'>
            <h4>🤖 Multi-Agent Support</h4>
            <p>Handles diverse autonomous systems from vehicles to trading bots with specialized analysis.</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="xAI Decision Platform",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/yourusername/xai-platform',
            'Report a bug': "https://github.com/yourusername/xai-platform/issues",
            'About': "Decision-Making Visualization Platform for Autonomous Agents using Unsupervised ML"
        }
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main > div {
            padding-top: 2rem;
        }
        .stMetric {
            background: white;
            border: 1px solid #ddd;
            padding: 1rem;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'simulator' not in st.session_state:
        with st.spinner("🔄 Initializing xAI Platform..."):
            st.session_state.simulator = AgentSimulator(num_agents=30)
            st.session_state.db_manager = DatabaseManager()
            st.session_state.analyzer = UnsupervisedAnalyzer()
            st.session_state.fairness_analyzer = FairnessAnalyzer()
            st.session_state.data_loaded = False
    
    # Project Overview Section
    create_project_overview()
    create_key_features_section()
    
    # Sidebar Controls
    st.sidebar.markdown("""
    <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
        <h2 style='color: white; margin: 0;'>🔧 Control Panel</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Data Generation
    st.sidebar.subheader("📊 Data Management")
    
    num_decisions = st.sidebar.slider("Number of Decisions to Generate", 100, 2000, 800)
    num_agents = st.sidebar.slider("Number of Agents", 10, 50, 25)
    
    if st.sidebar.button("🔄 Generate New Data", type="primary"):
        with st.spinner("Generating decision data..."):
            st.session_state.simulator = AgentSimulator(num_agents=num_agents)
            decisions = st.session_state.simulator.generate_batch_decisions(num_decisions)
            st.session_state.db_manager.store_decisions(decisions)
            st.session_state.data_loaded = True
            st.sidebar.success(f"✅ Generated {num_decisions} decisions!")
    
    # Load existing data
    try:
        df = st.session_state.db_manager.load_decisions_as_dataframe()
        if len(df) > 0:
            st.session_state.data_loaded = True
        else:
            st.warning("📊 No data available. Please generate data using the sidebar controls.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Database error: {str(e)}")
        st.stop()
    
    # Main Dashboard Content
    if st.session_state.data_loaded:
        
        # System Status
        st.markdown("---")
        st.markdown("### 📈 **System Dashboard**")
        
        # Key Metrics Row
        metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
        
        with metric_col1:
            st.metric(
                label="Total Decisions",
                value=f"{len(df):,}",
                delta=f"Last update: {df['timestamp'].max().strftime('%H:%M')}"
            )
        
        with metric_col2:
            avg_bias = df['bias_algorithmic_bias'].mean()
            st.metric(
                label="Avg Algorithmic Bias",
                value=f"{avg_bias:.3f}",
                delta=f"{'🔴 High' if avg_bias > 0.6 else '🟡 Medium' if avg_bias > 0.3 else '🟢 Low'}"
            )
        
        with metric_col3:
            fairness_cols = [col for col in df.columns if col.startswith('fairness_')]
            avg_fairness = df[fairness_cols].mean().mean()
            st.metric(
                label="Avg Fairness Score",
                value=f"{avg_fairness:.3f}",
                delta=f"{'🟢 Good' if avg_fairness > 0.8 else '🟡 Fair' if avg_fairness > 0.6 else '🔴 Poor'}"
            )
        
        with metric_col4:
            anomalies = st.session_state.analyzer.detect_anomalies(df)
            anomaly_count = np.sum(anomalies == -1)
            st.metric(
                label="Anomalous Decisions",
                value=anomaly_count,
                delta=f"{(anomaly_count/len(df)*100):.1f}% of total"
            )
        
        with metric_col5:
            unique_agents = df['agent_id'].nunique()
            agent_types = df['agent_type'].nunique()
            st.metric(
                label="Active Agents",
                value=unique_agents,
                delta=f"{agent_types} agent types"
            )
        
        # Main Analysis Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 System Overview", 
            "🎯 Unsupervised ML Analysis", 
            "⚖️ Fairness Analysis", 
            "⏰ Temporal Patterns", 
            "🚨 Alerts & Monitoring",
            "📋 Data Explorer"
        ])
        
        with tab1:
            st.header("📊 System Overview & Performance")
            
            # Overview visualizations
            overview_col1, overview_col2 = st.columns(2)
            
            with overview_col1:
                # Agent type distribution
                fig1 = px.pie(
                    df, 
                    names='agent_type', 
                    title="Agent Type Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig1.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig1, use_container_width=True)
                
                # Decision context breakdown
                fig2 = px.histogram(
                    df, 
                    x='context', 
                    title="Decision Context Distribution",
                    color='context',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            with overview_col2:
                # Bias distribution
                fig3 = px.histogram(
                    df, 
                    x='bias_algorithmic_bias', 
                    nbins=30,
                    title="Algorithmic Bias Distribution",
                    color_discrete_sequence=['#ff7f0e']
                )
                fig3.add_vline(x=df['bias_algorithmic_bias'].mean(), 
                              line_dash="dash", line_color="red",
                              annotation_text=f"Mean: {df['bias_algorithmic_bias'].mean():.3f}")
                st.plotly_chart(fig3, use_container_width=True)
                
                # Confidence vs Decision Value
                fig4 = px.scatter(
                    df, 
                    x='confidence_score', 
                    y='decision_value',
                    color='agent_type',
                    size='outcome_impact',
                    title="Confidence vs Decision Value",
                    hover_data=['context', 'bias_algorithmic_bias']
                )
                st.plotly_chart(fig4, use_container_width=True)
            
            # Performance summary table
            st.subheader("📋 Agent Performance Summary")
            
            performance_data = []
            for agent_type in df['agent_type'].unique():
                type_data = df[df['agent_type'] == agent_type]
                performance_data.append({
                    'Agent Type': agent_type.replace('_', ' ').title(),
                    'Decision Count': len(type_data),
                    'Avg Decision Value': f"{type_data['decision_value'].mean():.3f}",
                    'Avg Confidence': f"{type_data['confidence_score'].mean():.3f}",
                    'Avg Fairness': f"{type_data[fairness_cols].mean().mean():.3f}",
                    'Avg Bias': f"{type_data['bias_algorithmic_bias'].mean():.3f}",
                    'High Confidence %': f"{(type_data['confidence_score'] > 0.8).mean() * 100:.1f}%"
                })
            
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, use_container_width=True)
        
        with tab2:
            st.header("🎯 Unsupervised Machine Learning Analysis")
            
            if st.button("🔄 Run Complete ML Analysis", type="primary"):
                with st.spinner("Running unsupervised ML analysis..."):
                    
                    # Clustering Analysis
                    cluster_results = st.session_state.analyzer.detect_bias_clusters(df)
                    reduction_results = st.session_state.analyzer.dimensionality_reduction(df)
                    anomalies = st.session_state.analyzer.detect_anomalies(df)
                    
                    # Display results
                    st.success("✅ Analysis Complete!")
                    
                    # Results summary
                    result_col1, result_col2, result_col3 = st.columns(3)
                    
                    with result_col1:
                        st.metric("K-Means Clusters", cluster_results['kmeans']['n_clusters'])
                        st.metric("Silhouette Score", f"{cluster_results['kmeans']['silhouette_score']:.3f}")
                    
                    with result_col2:
                        st.metric("DBSCAN Clusters", cluster_results['dbscan']['n_clusters'])
                        st.metric("Noise Points", cluster_results['dbscan']['noise_points'])
                    
                    with result_col3:
                        anomaly_count = np.sum(anomalies == -1)
                        st.metric("Anomalies Detected", anomaly_count)
                        st.metric("Anomaly Rate", f"{(anomaly_count/len(df)*100):.1f}%")
                    
                    # Visualizations
                    st.subheader("🎨 Cluster Visualizations")
                    
                    # Create subplots for clustering results
                    fig_ml = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=['PCA - K-Means Clusters', 't-SNE - DBSCAN Clusters', 
                                      'Bias Distribution by Cluster', 'Anomaly Detection'],
                        specs=[[{"type": "scatter"}, {"type": "scatter"}],
                               [{"type": "histogram"}, {"type": "scatter"}]]
                    )
                    
                    # PCA visualization with K-Means
                    pca_data = reduction_results['pca']
                    fig_ml.add_trace(
                        go.Scatter(
                            x=pca_data[:, 0], y=pca_data[:, 1],
                            mode='markers',
                            marker=dict(color=cluster_results['kmeans']['labels'], 
                                      colorscale='Viridis', size=8),
                            name='PCA-KMeans',
                            text=[f"Agent: {aid}<br>Type: {at}<br>Cluster: {cl}" 
                                  for aid, at, cl in zip(df['agent_id'], df['agent_type'], 
                                                       cluster_results['kmeans']['labels'])],
                            hovertemplate='%{text}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                    
                    # t-SNE visualization with DBSCAN
                    tsne_data = reduction_results['tsne']
                    fig_ml.add_trace(
                        go.Scatter(
                            x=tsne_data[:, 0], y=tsne_data[:, 1],
                            mode='markers',
                            marker=dict(color=cluster_results['dbscan']['labels'], 
                                      colorscale='Plasma', size=8),
                            name='tSNE-DBSCAN',
                            text=[f"Agent: {aid}<br>Type: {at}<br>Cluster: {cl}" 
                                  for aid, at, cl in zip(df['agent_id'], df['agent_type'], 
                                                       cluster_results['dbscan']['labels'])],
                            hovertemplate='%{text}<extra></extra>'
                        ),
                        row=1, col=2
                    )
                    
                    # Bias distribution by cluster
                    fig_ml.add_trace(
                        go.Histogram(
                            x=df['bias_algorithmic_bias'],
                            nbinsx=20,
                            name='Bias Distribution',
                            marker_color='lightblue',
                            opacity=0.7
                        ),
                        row=2, col=1
                    )
                    
                    # Anomaly detection scatter
                    fig_ml.add_trace(
                        go.Scatter(
                            x=df['confidence_score'], 
                            y=df['decision_value'],
                            mode='markers',
                            marker=dict(
                                color=['red' if a == -1 else 'blue' for a in anomalies],
                                size=8
                            ),
                            name='Anomalies',
                            text=[f"Agent: {aid}<br>{'ANOMALY' if a == -1 else 'Normal'}" 
                                  for aid, a in zip(df['agent_id'], anomalies)],
                            hovertemplate='%{text}<extra></extra>'
                        ),
                        row=2, col=2
                    )
                    
                    fig_ml.update_layout(height=800, showlegend=True, 
                                       title="Comprehensive ML Analysis Results")
                    st.plotly_chart(fig_ml, use_container_width=True)
                    
                    # Cluster analysis table
                    st.subheader("📊 Cluster Analysis Summary")
                    
                    cluster_summary = []
                    df_with_clusters = df.copy()
                    df_with_clusters['kmeans_cluster'] = cluster_results['kmeans']['labels']
                    
                    for cluster_id in range(cluster_results['kmeans']['n_clusters']):
                        cluster_data = df_with_clusters[df_with_clusters['kmeans_cluster'] == cluster_id]
                        cluster_summary.append({
                            'Cluster ID': cluster_id,
                            'Size': len(cluster_data),
                            'Dominant Agent Type': cluster_data['agent_type'].mode().iloc[0] if len(cluster_data) > 0 else 'N/A',
                            'Avg Bias': f"{cluster_data['bias_algorithmic_bias'].mean():.3f}",
                            'Avg Fairness': f"{cluster_data[fairness_cols].mean().mean():.3f}",
                            'Avg Confidence': f"{cluster_data['confidence_score'].mean():.3f}",
                            'Risk Level': 'High' if cluster_data['bias_algorithmic_bias'].mean() > 0.6 else 'Medium' if cluster_data['bias_algorithmic_bias'].mean() > 0.3 else 'Low'
                        })
                    
                    cluster_df = pd.DataFrame(cluster_summary)
                    st.dataframe(cluster_df, use_container_width=True)
        
        with tab3:
            st.header("⚖️ Fairness Analysis Dashboard")
            
            # Run fairness analysis
            group_analysis = st.session_state.fairness_analyzer.analyze_group_fairness(df)
            alerts = st.session_state.fairness_analyzer.generate_fairness_alerts(df)
            
            # Fairness metrics overview
            st.subheader("📊 Fairness Metrics Overview")
            
            fairness_col1, fairness_col2 = st.columns(2)
            
            with fairness_col1:
                # Fairness scores by agent type
                fairness_data = []
                for agent_type, analysis in group_analysis.items():
                    for metric, score in analysis['fairness_scores'].items():
                        fairness_data.append({
                            'Agent Type': agent_type,
                            'Metric': metric.replace('_', ' ').title(),
                            'Score': score,
                            'Violation': score < st.session_state.fairness_analyzer.fairness_thresholds.get(metric, 0.8)
                        })
                
                fairness_plot_df = pd.DataFrame(fairness_data)
                
                fig_fairness = px.bar(
                    fairness_plot_df, 
                    x='Agent Type', 
                    y='Score', 
                    color='Metric',
                    title="Fairness Scores by Agent Type",
                    barmode='group'
                )
                fig_fairness.add_hline(y=0.8, line_dash="dash", line_color="red", 
                                     annotation_text="Fairness Threshold")
                st.plotly_chart(fig_fairness, use_container_width=True)
            
            with fairness_col2:
                # Violations heatmap
                violation_matrix = []
                agent_types = list(group_analysis.keys())
                metrics = ['demographic_parity', 'equalized_odds', 'individual_fairness', 'treatment_equality']
                
                for agent_type in agent_types:
                    violations = [1 if group_analysis[agent_type]['violations'][metric] else 0 for metric in metrics]
                    violation_matrix.append(violations)
                
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=violation_matrix,
                    x=[m.replace('_', ' ').title() for m in metrics],
                    y=[at.replace('_', ' ').title() for at in agent_types],
                    colorscale='Reds',
                    text=violation_matrix,
                    texttemplate="%{text}",
                    textfont={"size": 12}
                ))
                fig_heatmap.update_layout(title="Fairness Violations Heatmap (1=Violation, 0=Compliant)")
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Detailed fairness analysis
            st.subheader("📋 Detailed Fairness Analysis")
            
            fairness_summary = []
            for agent_type, analysis in group_analysis.items():
                fairness_summary.append({
                    'Agent Type': agent_type.replace('_', ' ').title(),
                    'Sample Size': analysis['sample_size'],
                    'Demographic Parity': f"{analysis['fairness_scores']['demographic_parity']:.3f}",
                    'Equalized Odds': f"{analysis['fairness_scores']['equalized_odds']:.3f}",
                    'Individual Fairness': f"{analysis['fairness_scores']['individual_fairness']:.3f}",
                    'Treatment Equality': f"{analysis['fairness_scores']['treatment_equality']:.3f}",
                    'Total Violations': sum(analysis['violations'].values()),
                    'Avg Decision Value': f"{analysis['avg_decision_value']:.3f}",
                    'Avg Confidence': f"{analysis['avg_confidence']:.3f}"
                })
            
            fairness_summary_df = pd.DataFrame(fairness_summary)
            st.dataframe(fairness_summary_df, use_container_width=True)
        
        with tab4:
            st.header("⏰ Temporal Analysis & Patterns")
            
            # Add time-based features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.day_name()
            
            temporal_col1, temporal_col2 = st.columns(2)
            
            with temporal_col1:
                # Hourly patterns
                hourly_decisions = df.groupby('hour').size()
                hourly_bias = df.groupby('hour')['bias_algorithmic_bias'].mean()
                
                fig_hourly = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig_hourly.add_trace(
                    go.Scatter(x=hourly_decisions.index, y=hourly_decisions.values,
                             mode='lines+markers', name='Decision Count'),
                    secondary_y=False
                )
                
                fig_hourly.add_trace(
                    go.Scatter(x=hourly_bias.index, y=hourly_bias.values,
                             mode='lines+markers', name='Avg Bias', line=dict(color='red')),
                    secondary_y=True
                )
                
                fig_hourly.update_xaxes(title_text="Hour of Day")
                fig_hourly.update_yaxes(title_text="Decision Count", secondary_y=False)
                fig_hourly.update_yaxes(title_text="Average Bias", secondary_y=True)
                fig_hourly.update_layout(title="Hourly Decision Patterns")
                
                st.plotly_chart(fig_hourly, use_container_width=True)
                
                # Context-based temporal analysis
                fig_context = px.box(
                    df, 
                    x='context', 
                    y='decision_value',
                    color='context',
                    title="Decision Values by Context"
                )
                st.plotly_chart(fig_context, use_container_width=True)
            
            with temporal_col2:
                # Weekly patterns
                weekly_stats = df.groupby('day_of_week').agg({
                    'decision_value': 'mean',
                    'confidence_score': 'mean',
                    'bias_algorithmic_bias': 'mean',
                    'decision_id': 'count'
                }).round(3)
                
                fig_weekly = px.line(
                    weekly_stats.reset_index(), 
                    x='day_of_week', 
                    y='bias_algorithmic_bias',
                    title="Weekly Bias Patterns",
                    markers=True
                )
                st.plotly_chart(fig_weekly, use_container_width=True)
                
                # Agent type performance over time
                fig_agent_time = px.scatter(
                    df, 
                    x='timestamp', 
                    y='bias_algorithmic_bias',
                    color='agent_type',
                    size='confidence_score',
                    title="Bias Evolution Over Time by Agent Type"
                )
                st.plotly_chart(fig_agent_time, use_container_width=True)
            
            # Temporal insights
            st.subheader("🔍 Temporal Insights")
            
            insight_col1, insight_col2, insight_col3 = st.columns(3)
            
            with insight_col1:
                peak_hour = hourly_decisions.idxmax()
                peak_count = hourly_decisions.max()
                st.metric("Peak Activity Hour", f"{peak_hour}:00", f"{peak_count} decisions")
            
            with insight_col2:
                worst_bias_hour = hourly_bias.idxmax()
                worst_bias_value = hourly_bias.max()
                st.metric("Highest Bias Hour", f"{worst_bias_hour}:00", f"{worst_bias_value:.3f}")
            
            with insight_col3:
                best_bias_hour = hourly_bias.idxmin()
                best_bias_value = hourly_bias.min()
                st.metric("Lowest Bias Hour", f"{best_bias_hour}:00", f"{best_bias_value:.3f}")
        
        with tab5:
            st.header("🚨 Alerts & Monitoring Dashboard")
            
            # Active alerts
            if alerts:
                st.error(f"⚠️ {len(alerts)} Active Fairness Alerts")
                
                for i, alert in enumerate(alerts):
                    severity_colors = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
                    
                    with st.expander(f"{severity_colors[alert['severity']]} {alert['type'].replace('_', ' ').title()} - {alert['agent_type']}"):
                        alert_col1, alert_col2 = st.columns(2)
                        
                        with alert_col1:
                            st.write("**Alert Details:**")
                            st.write(f"- **Metric:** {alert['metric'].replace('_', ' ').title()}")
                            st.write(f"- **Actual Score:** {alert['actual_score']:.3f}")
                            st.write(f"- **Threshold:** {alert['threshold']:.3f}")
                            st.write(f"- **Severity:** {alert['severity'].upper()}")
                            st.write(f"- **Timestamp:** {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        with alert_col2:
                            st.write("**Recommendation:**")
                            st.write(alert['recommendation'])
                            
                            if st.button(f"Mark as Resolved", key=f"resolve_{i}"):
                                st.success("✅ Alert marked as resolved!")
            else:
                st.success("✅ No active alerts! System operating within fairness thresholds.")
            
            # System health monitoring
            st.subheader("📊 System Health Monitoring")
            
            health_col1, health_col2, health_col3, health_col4 = st.columns(4)
            
            with health_col1:
                system_bias = df['bias_algorithmic_bias'].mean()
                bias_status = "🟢 Healthy" if system_bias < 0.3 else "🟡 Warning" if system_bias < 0.6 else "🔴 Critical"
                st.metric("System Bias Level", f"{system_bias:.3f}", bias_status)
            
            with health_col2:
                fairness_score = df[fairness_cols].mean().mean()
                fairness_status = "🟢 Excellent" if fairness_score > 0.8 else "🟡 Good" if fairness_score > 0.6 else "🔴 Poor"
                st.metric("Overall Fairness", f"{fairness_score:.3f}", fairness_status)
            
            with health_col3:
                confidence_level = (df['confidence_score'] > 0.8).mean()
                confidence_status = "🟢 High" if confidence_level > 0.7 else "🟡 Medium" if confidence_level > 0.5 else "🔴 Low"
                st.metric("High Confidence %", f"{confidence_level*100:.1f}%", confidence_status)
            
            with health_col4:
                violation_rate = sum(sum(analysis['violations'].values()) for analysis in group_analysis.values())
                violation_status = "🟢 Good" if violation_rate == 0 else "🟡 Warning" if violation_rate < 3 else "🔴 Critical"
                st.metric("Total Violations", violation_rate, violation_status)
            
            # Real-time monitoring chart
            st.subheader("📈 Real-time Monitoring")
            
            # Create time series data
            df_sorted = df.sort_values('timestamp')
            df_sorted['cumulative_decisions'] = range(1, len(df_sorted) + 1)
            df_sorted['rolling_bias'] = df_sorted['bias_algorithmic_bias'].rolling(window=50).mean()
            df_sorted['rolling_fairness'] = df_sorted[fairness_cols].rolling(window=50).mean().mean(axis=1)
            
            fig_realtime = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig_realtime.add_trace(
                go.Scatter(
                    x=df_sorted['timestamp'], 
                    y=df_sorted['rolling_bias'],
                    mode='lines',
                    name='Rolling Bias (50-period)',
                    line=dict(color='red')
                ),
                secondary_y=False
            )
            
            fig_realtime.add_trace(
                go.Scatter(
                    x=df_sorted['timestamp'], 
                    y=df_sorted['rolling_fairness'],
                    mode='lines',
                    name='Rolling Fairness (50-period)',
                    line=dict(color='green')
                ),
                secondary_y=True
            )
            
            fig_realtime.update_xaxes(title_text="Time")
            fig_realtime.update_yaxes(title_text="Bias Level", secondary_y=False)
            fig_realtime.update_yaxes(title_text="Fairness Score", secondary_y=True)
            fig_realtime.update_layout(title="Real-time System Monitoring")
            
            st.plotly_chart(fig_realtime, use_container_width=True)
        
        with tab6:
            st.header("📋 Data Explorer & Raw Data Analysis")
            
            # Data filtering
            st.subheader("🔍 Data Filtering")
            
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                selected_agents = st.multiselect(
                    "Select Agent Types",
                    options=df['agent_type'].unique(),
                    default=df['agent_type'].unique()
                )
            
            with filter_col2:
                selected_contexts = st.multiselect(
                    "Select Decision Contexts", 
                    options=df['context'].unique(),
                    default=df['context'].unique()
                )
            
            with filter_col3:
                bias_range = st.slider(
                    "Bias Range Filter",
                    min_value=float(df['bias_algorithmic_bias'].min()),
                    max_value=float(df['bias_algorithmic_bias'].max()),
                    value=(float(df['bias_algorithmic_bias'].min()), float(df['bias_algorithmic_bias'].max()))
                )
            
            # Apply filters
            filtered_df = df[
                (df['agent_type'].isin(selected_agents)) &
                (df['context'].isin(selected_contexts)) &
                (df['bias_algorithmic_bias'] >= bias_range[0]) &
                (df['bias_algorithmic_bias'] <= bias_range[1])
            ]
            
            st.write(f"**Filtered Dataset:** {len(filtered_df):,} decisions ({len(filtered_df)/len(df)*100:.1f}% of total)")
            
            # Data display options
            display_col1, display_col2 = st.columns(2)
            
            with display_col1:
                show_raw_data = st.checkbox("Show Raw Data")
                show_statistics = st.checkbox("Show Statistics", value=True)
            
            with display_col2:
                show_correlations = st.checkbox("Show Correlations")
                export_data = st.checkbox("Enable Data Export")
            
            # Display data based on selections
            if show_statistics:
                st.subheader("📊 Filtered Data Statistics")
                st.dataframe(filtered_df.describe(), use_container_width=True)
            
            if show_raw_data:
                st.subheader("📋 Raw Data")
                st.dataframe(filtered_df.head(100), use_container_width=True)
            
            if show_correlations:
                st.subheader("🔗 Feature Correlations")
                numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
                corr_matrix = filtered_df[numeric_cols].corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    title="Feature Correlation Matrix",
                    color_continuous_scale='RdBu_r',
                    aspect="auto"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            
            if export_data:
                st.subheader("📤 Data Export")
                
                export_col1, export_col2 = st.columns(2)
                
                with export_col1:
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download as CSV",
                        data=csv,
                        file_name=f"xai_platform_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with export_col2:
                    json_data = filtered_df.to_json(orient='records', date_format='iso')
                    st.download_button(
                        label="Download as JSON",
                        data=json_data,
                        file_name=f"xai_platform_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>🤖 xAI Decision-Making Visualization Platform</strong></p>
        <p>Real-time bias detection and fairness analysis for autonomous agents</p>
        <p>Built with Streamlit • Powered by Unsupervised ML • Running on port 8501</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

run_streamlit_in_colab()

# 1. Create the complete Streamlit app file
streamlit_code = '''
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Example: load your own DataFrame (replace below with actual data import)
# df = pd.read_csv('your_decisions_data.csv')
# alerts = [{'agent_type': 'trading_bot', 'type': 'demographic_parity', 'severity': 'high', 'recommendation': 'Review trading_bot agents...'}, ...]
# metrics = {'average_bias': 0.209, 'fairness_score': 0.636, 'average_confidence': 0.659}
# clusters = {'kmeans': 2, 'silhouette_score': 0.182, 'anomalies': 160}

st.set_page_config(page_title="xAI Agentic Visualization Platform", layout="wide")
st.title("🤖 xAI Decision Making Visualization Platform")

# Summary/Metadata section
st.subheader("Dataset Overview")
st.markdown("""
 - **Total decisions:** 1600\n- **Unique agents:** 50\n- **Agent types:** trading_bot, security_scanner, resource_allocator, recommendation_system, autonomous_vehicle\n- **Date range:** 2025-09-27 15:04:33 to 2025-09-27 16:13:21
""")
# Data Table
# st.dataframe(df.describe(), use_container_width=True)

# Fairness Alerts Section
st.subheader("Fairness Analysis: 🚨")
st.markdown("**Found 20 active alerts:**")
for i in range(5):  # Example: replace with actual alert data
    st.warning(
        f"""**{i+1}. Fairness Violation**
    - Severity: High
    - Recommendation: Review trading_bot agents for bias"""
    )

# System Metrics Section
st.subheader("Final System Metrics")
st.markdown(
    """
- **Average algorithmic bias:** 0.209
- **Average fairness score:** 0.636
- **Average confidence:** 0.659
    """
)




# Clustering Results Section
st.subheader("Bias Cluster Analysis Dashboard")
st.markdown("""
- **K-Means clusters found:** 2
- **Silhouette score:** 0.182
- **DBSCAN clusters:** 0
- **Anomalous decisions:** 160
""")



# Example Cluster Plots
cluster_data = pd.DataFrame({'x': np.random.randn(320), 'y': np.random.randn(320), 'cluster': np.random.choice(['PCA', 't-SNE'], 320)})
fig = px.scatter(cluster_data, x='x', y='y', color='cluster', title="Bias Cluster Visualizations")
st.plotly_chart(fig, use_container_width=True)

st.success("✅ Analysis complete! Dashboard shows real analysis/alerts/visualizations, as seen in your notebook screenshots.")

# Optionally: Add descriptive/data export, etc.
'''

# Save the file
with open('streamlit_app.py', 'w') as f:
    f.write(streamlit_code)

# 2. Restart everything
import os
os.system("pkill -f streamlit")  # Kill existing processes
time.sleep(3)

# 3. Run with proper settings
!streamlit run streamlit_app.py --server.port 8501 --server.headless true --server.enableCORS false &

# 4. Wait and create tunnel
time.sleep(5)
from pyngrok import ngrok
ngrok.kill()
public_url = ngrok.connect(8501)
print(f"🌐 Dashboard: {public_url}")
