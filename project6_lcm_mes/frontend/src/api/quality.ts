import api from './client'

export interface DefectRecord {
  id: number
  product_code: string
  stage: number
  defect_type: string
  description: string
  severity: string
  status: string
}

export interface Inspection {
  id: number
  product_code: string
  type: string
  stage: number
  result: string
  inspector_id?: number
}

export const qualityApi = {
  getDefects: () => api.get<DefectRecord[]>('/api/v1/quality/defects'),
  createDefect: (data: Partial<DefectRecord>) => api.post('/api/v1/quality/defects', data),
  getInspections: () => api.get<Inspection[]>('/api/v1/quality/inspections'),
  createInspection: (data: Partial<Inspection>) => api.post('/api/v1/quality/inspections', data)
}
