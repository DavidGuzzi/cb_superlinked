import pandas as pd
from typing import Dict, Any
from scipy import stats
import numpy as np

class ABTestingAnalytics:
    """Analizador estadístico para experimentos AB Testing"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def analyze_ab_test_results(self) -> Dict[str, Any]:
        """Analiza los resultados completos del AB Testing"""
        control_data = self.data[self.data['experimento'] == 'Control']
        
        # Obtener todos los grupos de experimento
        experiment_groups = self.data[self.data['experimento'] != 'Control']['experimento'].unique()
        
        results = {'control': self._calculate_metrics(control_data)}
        
        # Analizar cada experimento vs control
        for experiment_name in experiment_groups:
            experiment_data = self.data[self.data['experimento'] == experiment_name]
            experiment_metrics = self._calculate_metrics(experiment_data)
            
            # Calcular lifts
            conversion_lift = self._calculate_lift(
                experiment_metrics['avg_conversion_rate'],
                results['control']['avg_conversion_rate']
            )
            
            revenue_lift = self._calculate_lift(
                experiment_metrics['total_revenue'],
                results['control']['total_revenue']
            )
            
            # Test de significancia estadística
            statistical_test = self._perform_statistical_test(control_data, experiment_data)
            
            results[experiment_name] = {
                'metrics': experiment_metrics,
                'conversion_lift': conversion_lift,
                'revenue_lift': revenue_lift,
                'statistical_significance': statistical_test
            }
        
        # Análisis por segmentos (usando todos los datos)
        regional_analysis = self._analyze_by_region()
        store_type_analysis = self._analyze_by_store_type()
        
        results.update({
            'regional_analysis': regional_analysis,
            'store_type_analysis': store_type_analysis,
            'summary': self._generate_summary_multi(results)
        })
        
        return results
    
    def _calculate_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calcula métricas para un grupo de datos"""
        return {
            'total_usuarios': data['usuarios'].sum(),
            'total_conversiones': data['conversiones'].sum(),
            'total_revenue': data['revenue'].sum(),
            'avg_conversion_rate': data['conversion_rate'].mean(),
            'median_conversion_rate': data['conversion_rate'].median(),
            'std_conversion_rate': data['conversion_rate'].std(),
            'avg_revenue_per_user': data['revenue'].sum() / data['usuarios'].sum(),
            'avg_revenue_per_conversion': data['revenue'].sum() / data['conversiones'].sum() if data['conversiones'].sum() > 0 else 0
        }
    
    def _calculate_lift(self, experiment_value: float, control_value: float) -> float:
        """Calcula el lift porcentual"""
        if control_value == 0:
            return 0
        return ((experiment_value - control_value) / control_value) * 100
    
    def _perform_statistical_test(self, control_data: pd.DataFrame, 
                                 experiment_data: pd.DataFrame) -> Dict[str, Any]:
        """Realiza test de significancia estadística"""
        
        # T-test para conversion rates
        control_rates = control_data['conversion_rate'].values
        experiment_rates = experiment_data['conversion_rate'].values
        
        t_stat, p_value = stats.ttest_ind(control_rates, experiment_rates)
        
        # Chi-square test para proporciones
        control_conversions = control_data['conversiones'].sum()
        control_users = control_data['usuarios'].sum()
        experiment_conversions = experiment_data['conversiones'].sum()
        experiment_users = experiment_data['usuarios'].sum()
        
        # Tabla de contingencia para chi-square
        observed = np.array([
            [control_conversions, control_users - control_conversions],
            [experiment_conversions, experiment_users - experiment_conversions]
        ])
        
        chi2, chi2_p_value, dof, expected = stats.chi2_contingency(observed)
        
        return {
            't_test': {
                't_statistic': t_stat,
                'p_value': p_value,
                'is_significant': p_value < 0.05,
                'confidence_level': 0.95
            },
            'chi_square_test': {
                'chi2_statistic': chi2,
                'p_value': chi2_p_value,
                'degrees_of_freedom': dof,
                'is_significant': chi2_p_value < 0.05
            },
            'overall_significance': p_value < 0.05 and chi2_p_value < 0.05
        }
    
    def _analyze_by_region(self) -> Dict[str, Dict]:
        """Analiza resultados por región"""
        regional_results = {}
        
        for region in self.data['region'].unique():
            region_data = self.data[self.data['region'] == region]
            control_region = region_data[region_data['experimento'] == 'Control']
            experiment_region = region_data[region_data['experimento'] == 'Experimento_A']
            
            if len(control_region) > 0 and len(experiment_region) > 0:
                control_cr = control_region['conversion_rate'].mean()
                experiment_cr = experiment_region['conversion_rate'].mean()
                
                regional_results[region] = {
                    'control_conversion_rate': control_cr,
                    'experiment_conversion_rate': experiment_cr,
                    'lift': self._calculate_lift(experiment_cr, control_cr),
                    'control_revenue': control_region['revenue'].sum(),
                    'experiment_revenue': experiment_region['revenue'].sum(),
                    'sample_size_control': len(control_region),
                    'sample_size_experiment': len(experiment_region)
                }
        
        return regional_results
    
    def _analyze_by_store_type(self) -> Dict[str, Dict]:
        """Analiza resultados por tipo de tienda"""
        store_type_results = {}
        
        for store_type in self.data['tipo_tienda'].unique():
            store_data = self.data[self.data['tipo_tienda'] == store_type]
            control_store = store_data[store_data['experimento'] == 'Control']
            experiment_store = store_data[store_data['experimento'] == 'Experimento_A']
            
            if len(control_store) > 0 and len(experiment_store) > 0:
                control_cr = control_store['conversion_rate'].mean()
                experiment_cr = experiment_store['conversion_rate'].mean()
                
                store_type_results[store_type] = {
                    'control_conversion_rate': control_cr,
                    'experiment_conversion_rate': experiment_cr,
                    'lift': self._calculate_lift(experiment_cr, control_cr),
                    'control_revenue': control_store['revenue'].sum(),
                    'experiment_revenue': experiment_store['revenue'].sum(),
                    'sample_size_control': len(control_store),
                    'sample_size_experiment': len(experiment_store)
                }
        
        return store_type_results
    
    def _generate_summary(self, control_metrics: Dict, experiment_metrics: Dict,
                         conversion_lift: float, revenue_lift: float,
                         statistical_test: Dict) -> str:
        """Genera un resumen ejecutivo del análisis"""
        
        significance_text = "estadísticamente significativo" if statistical_test['overall_significance'] else "no estadísticamente significativo"
        
        summary = f"""
        RESUMEN EJECUTIVO DEL AB TESTING:
        
        📊 MÉTRICAS PRINCIPALES:
        • Control: {control_metrics['total_conversiones']} conversiones de {control_metrics['total_usuarios']} usuarios ({control_metrics['avg_conversion_rate']:.2f}% CR)
        • Experimento A: {experiment_metrics['total_conversiones']} conversiones de {experiment_metrics['total_usuarios']} usuarios ({experiment_metrics['avg_conversion_rate']:.2f}% CR)
        
        📈 RESULTADOS:
        • Lift en conversión: {conversion_lift:+.2f}%
        • Lift en revenue: {revenue_lift:+.2f}%
        • Resultado: {significance_text}
        • P-value (t-test): {statistical_test['t_test']['p_value']:.4f}
        
        💰 IMPACTO FINANCIERO:
        • Revenue Control: ${control_metrics['total_revenue']:,.2f}
        • Revenue Experimento: ${experiment_metrics['total_revenue']:,.2f}
        • Diferencia: ${experiment_metrics['total_revenue'] - control_metrics['total_revenue']:+,.2f}
        """
        
        return summary.strip()
    
    def _generate_summary_multi(self, results: Dict) -> str:
        """Genera un resumen ejecutivo para múltiples experimentos"""
        
        control = results['control']
        summary = f"""
        RESUMEN EJECUTIVO DEL AB TESTING MULTI-VARIANTE:
        
        📊 GRUPO CONTROL:
        • {control['total_conversiones']} conversiones de {control['total_usuarios']} usuarios
        • Conversion Rate: {control['avg_conversion_rate']:.2f}%
        • Revenue: ${control['total_revenue']:,.2f}
        
        📈 RESULTADOS POR EXPERIMENTO:"""
        
        experiment_groups = [key for key in results.keys() if key not in ['control', 'regional_analysis', 'store_type_analysis', 'summary']]
        
        for exp_name in experiment_groups:
            exp_data = results[exp_name]
            metrics = exp_data['metrics']
            significance = "✅ Significativo" if exp_data['statistical_significance']['overall_significance'] else "❌ No significativo"
            
            summary += f"""
        
        • {exp_name}:
          - CR: {metrics['avg_conversion_rate']:.2f}% (Lift: {exp_data['conversion_lift']:+.2f}%)
          - Revenue: ${metrics['total_revenue']:,.2f} (Lift: {exp_data['revenue_lift']:+.2f}%)
          - {significance} (p-value: {exp_data['statistical_significance']['t_test']['p_value']:.4f})"""
        
        # Encontrar el mejor experimento
        best_experiment = max(experiment_groups, 
                            key=lambda x: results[x]['conversion_lift'])
        best_lift = results[best_experiment]['conversion_lift']
        
        summary += f"""
        
        🏆 MEJOR EXPERIMENTO: {best_experiment} (Lift: {best_lift:+.2f}%)
        """
        
        return summary.strip()