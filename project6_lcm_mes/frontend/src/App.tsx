import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from 'react-query'
import { ConfigProvider } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import { AuthProvider, useAuth } from './contexts/AuthContext'
import MainLayout from './layouts/MainLayout'
import LoginPage from './pages/LoginPage'
import EngineeringDashboard from './pages/engineering/DashboardPage'
import EquipmentPage from './pages/engineering/EquipmentPage'
import OperatorDashboard from './pages/production/OperatorDashboard'
import WorkOrderPage from './pages/production/WorkOrderPage'
import RepairPage from './pages/production/RepairPage'
import PMCDashboard from './pages/pmc/PMCDashboard'
import OrdersPage from './pages/pmc/OrdersPage'
import MaterialPage from './pages/pmc/MaterialPage'
import QualityDashboard from './pages/quality/QualityDashboard'
import InspectionPage from './pages/quality/InspectionPage'
import DefectsPage from './pages/quality/DefectsPage'
import WarehouseDashboard from './pages/warehouse/WarehouseDashboard'
import IncomingPage from './pages/warehouse/IncomingPage'
import ShippingPage from './pages/warehouse/ShippingPage'
import AIDashboardPage from './pages/ai/AIDashboardPage'
import DashboardScreen from './pages/screen/DashboardScreen'

const queryClient = new QueryClient()

const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
  const { isAuthenticated } = useAuth()
  return isAuthenticated ? <>{children}</> : <Navigate to="/login" replace />
}

const RoleRoute = ({ children, allowedRoles }: { children: React.ReactNode, allowedRoles: string[] }) => {
  const { user } = useAuth()
  return user && allowedRoles.includes(user.role) ? <>{children}</> : <Navigate to="/" replace />
}

const AppContent = () => {
  const { user } = useAuth()

  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      
      <Route path="/screen" element={
        <ProtectedRoute>
          <DashboardScreen />
        </ProtectedRoute>
      } />

      <Route path="/" element={
        <ProtectedRoute>
          <MainLayout />
        </ProtectedRoute>
      }>
        <Route index element={
          user?.role === 'admin' ? <EngineeringDashboard /> :
          user?.role === 'manager' ? <PMCDashboard /> :
          user?.role === 'operator' ? <OperatorDashboard /> :
          user?.role === 'inspector' ? <QualityDashboard /> :
          user?.role === 'warehouse' ? <WarehouseDashboard /> :
          <EngineeringDashboard />
        } />

        <Route path="equipment" element={
          <RoleRoute allowedRoles={['admin']}>
            <EquipmentPage />
          </RoleRoute>
        } />

        <Route path="work-orders" element={
          <RoleRoute allowedRoles={['admin', 'operator', 'manager']}>
            <WorkOrderPage />
          </RoleRoute>
        } />

        <Route path="repair" element={
          <RoleRoute allowedRoles={['admin', 'operator', 'manager']}>
            <RepairPage />
          </RoleRoute>
        } />

        <Route path="orders" element={
          <RoleRoute allowedRoles={['admin', 'manager']}>
            <OrdersPage />
          </RoleRoute>
        } />

        <Route path="material" element={
          <RoleRoute allowedRoles={['admin', 'manager']}>
            <MaterialPage />
          </RoleRoute>
        } />

        <Route path="quality" element={
          <RoleRoute allowedRoles={['admin', 'inspector']}>
            <QualityDashboard />
          </RoleRoute>
        } />

        <Route path="inspection" element={
          <RoleRoute allowedRoles={['admin', 'inspector']}>
            <InspectionPage />
          </RoleRoute>
        } />

        <Route path="defects" element={
          <RoleRoute allowedRoles={['admin', 'inspector']}>
            <DefectsPage />
          </RoleRoute>
        } />

        <Route path="incoming" element={
          <RoleRoute allowedRoles={['admin', 'warehouse']}>
            <IncomingPage />
          </RoleRoute>
        } />

        <Route path="shipping" element={
          <RoleRoute allowedRoles={['admin', 'warehouse']}>
            <ShippingPage />
          </RoleRoute>
        } />

        <Route path="ai" element={
          <RoleRoute allowedRoles={['admin', 'manager', 'inspector']}>
            <AIDashboardPage />
          </RoleRoute>
        } />
      </Route>
    </Routes>
  )
}

const App = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <ConfigProvider locale={zhCN}>
        <AuthProvider>
          <BrowserRouter>
            <AppContent />
          </BrowserRouter>
        </AuthProvider>
      </ConfigProvider>
    </QueryClientProvider>
  )
}

export default App