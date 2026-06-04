import api from './client'

export interface User {
  id: number
  username: string
  email: string
  role: 'admin' | 'manager' | 'operator' | 'inspector' | 'warehouse'
  department: string
  name: string
}

export interface LoginResponse {
  access_token: string
  token_type: string
  user: User
}

export const authApi = {
  login: (username: string, password: string) =>
    api.post<LoginResponse>('/api/v1/auth/login', { username, password }),
  getCurrentUser: () => api.get<User>('/api/v1/auth/me')
}
