import api from './client'

export interface DefectPredictionRequest {
  product_code: string
  process_step: string
  equipment_id?: number
  operator_id?: number
  environmental_data?: Record<string, number>
}

export interface DefectPredictionResponse {
  product_code: string
  defect_probability: number
  defect_type?: string
  confidence: number
  recommendations: string[]
}

export interface EquipmentHealthRequest {
  equipment_id: number
  look_ahead_days?: number
}

export interface EquipmentHealthResponse {
  equipment_id: number
  equipment_name: string
  health_score: number
  risk_level: string
  predicted_failure_date?: string
  recommended_actions: string[]
}

export interface RepairDecisionRequest {
  product_code: string
  defect_type: string
  defect_severity: string
  repair_history?: Record<string, unknown>[]
}

export interface RepairDecisionResponse {
  product_code: string
  should_repair: boolean
  repair_cost_estimate: number
  repair_time_estimate: number
  alternative_action?: string
  confidence: number
}

export interface YieldPredictionRequest {
  product_type: string
  batch_size: number
  process_parameters?: Record<string, number>
}

export interface YieldPredictionResponse {
  product_type: string
  predicted_yield: number
  confidence: number
  key_factors: string[]
  optimization_suggestions: string[]
}

export interface SchedulingOptimizationRequest {
  order_priority?: string
  resource_constraints?: Record<string, unknown>
}

export interface SchedulingOptimizationResponse {
  optimized_schedule: Record<string, unknown>[]
  expected_throughput: number
  bottleneck_analysis: string[]
}

export interface DefectTrendData {
  hour: string
  defect_count: number
  defect_rate: number
}

export interface DefectTrendsResponse {
  trend_data: DefectTrendData[]
  top_defects: { type: string; count: number; rate: number }[]
  overall_defect_rate: number
  anomaly_detection: { time: string; type: string; value: string }[]
}

export interface QualityReportResponse {
  report_period: string
  total_units: number
  good_units: number
  defect_units: number
  yield_rate: number
  defect_distribution: { type: string; count: number; percentage: number }[]
  recommendations: string[]
  ai_summary: string
}

export interface AIDashboardResponse {
  defect_trends: DefectTrendsResponse
  quality_report: QualityReportResponse
  yield_forecast: {
    today: YieldPredictionResponse
    week: { predicted_yield: number; trend: string }
  }
  equipment_alerts: EquipmentHealthResponse[]
}

export const aiApi = {
  predictDefect: (data: DefectPredictionRequest) =>
    api.post<DefectPredictionResponse>('/api/v1/ai/defect-prediction', data),

  getEquipmentHealth: (data: EquipmentHealthRequest) =>
    api.post<EquipmentHealthResponse>('/api/v1/ai/equipment-health', data),

  makeRepairDecision: (data: RepairDecisionRequest) =>
    api.post<RepairDecisionResponse>('/api/v1/ai/repair-decision', data),

  predictYield: (data: YieldPredictionRequest) =>
    api.post<YieldPredictionResponse>('/api/v1/ai/yield-prediction', data),

  optimizeScheduling: (data?: SchedulingOptimizationRequest) =>
    api.post<SchedulingOptimizationResponse>('/api/v1/ai/scheduling-optimization', data || {}),

  getDefectTrends: (hours?: number) =>
    api.get<DefectTrendsResponse>('/api/v1/ai/defect-trends', { params: { hours } }),

  getQualityReport: (productType?: string) =>
    api.get<QualityReportResponse>('/api/v1/ai/quality-report', { params: { product_type: productType } }),

  getAIDashboard: () => api.get<AIDashboardResponse>('/api/v1/ai/dashboard')
}