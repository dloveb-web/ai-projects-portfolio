import api from './client'

export interface Material {
  id: number
  code: string
  name: string
  type: string
  quantity: number
  unit: string
}

export interface MaterialLot {
  id: number
  lot_number: string
  material_id: number
  quantity: number
  status: string
}

export const materialApi = {
  getMaterials: () => api.get<Material[]>('/api/v1/material/list'),
  getLots: () => api.get<MaterialLot[]>('/api/v1/material/lots')
}
