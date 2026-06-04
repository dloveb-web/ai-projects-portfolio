import api from './client'

export interface ProductionOrder {
  id: number
  order_number: string
  product_name: string
  quantity: number
  status: string
  priority: number
  start_date?: string
  end_date?: string
}

export interface WorkOrder {
  id: number
  work_order_number: string
  production_order_id: number
  product_code: string
  stage: number
  status: string
  current_product_code?: string
  quantity: number
  start_time?: string
  end_time?: string
}

export interface ProductTrace {
  id: number
  product_code: string
  stage: number
  status: string
  stage_1_code?: string
  stage_2_code?: string
  stage_3_code?: string
  stage_4_code?: string
  final_code?: string
}

export const productionApi = {
  getOrders: () => api.get<ProductionOrder[]>('/api/v1/production/orders'),
  createOrder: (data: Partial<ProductionOrder>) => api.post('/api/v1/production/orders', data),
  getWorkOrders: () => api.get<WorkOrder[]>('/api/v1/production/work-orders'),
  createWorkOrder: (data: Partial<WorkOrder>) => api.post('/api/v1/production/work-orders', data),
  traceProduct: (code: string) => api.get<ProductTrace>(`/api/v1/production/trace/${code}`),
  getDashboard: () => api.get('/api/v1/reports/dashboard')
}
