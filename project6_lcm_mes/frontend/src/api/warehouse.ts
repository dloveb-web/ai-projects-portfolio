import api from './client'

export interface Box {
  id: number
  box_number: string
  quantity: number
  status: string
}

export interface Pallet {
  id: number
  pallet_number: string
  status: string
}

export interface Shipment {
  id: number
  shipment_number: string
  status: string
}

export const warehouseApi = {
  getBoxes: () => api.get<Box[]>('/api/v1/warehouse/boxes'),
  getPallets: () => api.get<Pallet[]>('/api/v1/warehouse/pallets'),
  getShipments: () => api.get<Shipment[]>('/api/v1/warehouse/shipments'),
  createBox: (data: any) => api.post('/api/v1/warehouse/boxes', data),
  createPallet: (data: any) => api.post('/api/v1/warehouse/pallets', data)
}
