import api from './client'

export interface Equipment {
  id: number
  code: string
  name: string
  type: string
  model: string
  status: string
  location: string
  health_score: number
}

export interface MaintenanceRecord {
  id: number
  equipment_id: number
  type: string
  description: string
  status: string
  start_time?: string
  end_time?: string
}

export const equipmentApi = {
  getEquipment: () => api.get<Equipment[]>('/api/v1/equipment/list'),
  getStatus: () => api.get('/api/v1/equipment/status'),
  getMaintenance: () => api.get<MaintenanceRecord[]>('/api/v1/equipment/maintenance')
}
