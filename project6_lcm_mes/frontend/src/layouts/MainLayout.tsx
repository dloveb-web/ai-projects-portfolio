import { Layout, Menu, Avatar, Dropdown, Badge, Space } from 'antd'
import { 
  DashboardOutlined, 
  DesktopOutlined, 
  ShopOutlined, 
  BarChartOutlined,
  InboxOutlined,
  LogoutOutlined,
  UserOutlined,
  RocketOutlined
} from '@ant-design/icons'
import { Outlet, useNavigate, useLocation } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'

const { Header, Sider, Content } = Layout

const menuItems: Record<string, any[]> = {
  admin: [
    { key: '/', icon: <DashboardOutlined />, label: '全局仪表盘' },
    { key: '/equipment', icon: <DesktopOutlined />, label: '设备管理' },
    { key: '/quality', icon: <BarChartOutlined />, label: '质量监控' },
    { key: '/ai', icon: <RocketOutlined />, label: 'AI智能分析' }
  ],
  manager: [
    { key: '/', icon: <DashboardOutlined />, label: '生产计划' },
    { key: '/orders', icon: <ShopOutlined />, label: '订单管理' },
    { key: '/material', icon: <InboxOutlined />, label: '物料状态' },
    { key: '/ai', icon: <RocketOutlined />, label: 'AI智能分析' }
  ],
  operator: [
    { key: '/', icon: <DashboardOutlined />, label: '工单看板' },
    { key: '/work-orders', icon: <ShopOutlined />, label: '工单操作' },
    { key: '/repair', icon: <DesktopOutlined />, label: '不良品维修' }
  ],
  inspector: [
    { key: '/', icon: <DashboardOutlined />, label: '质量监控' },
    { key: '/inspection', icon: <BarChartOutlined />, label: '检测记录' },
    { key: '/defects', icon: <DesktopOutlined />, label: '不良品管理' },
    { key: '/ai', icon: <RocketOutlined />, label: 'AI智能分析' }
  ],
  warehouse: [
    { key: '/', icon: <DashboardOutlined />, label: '库存看板' },
    { key: '/incoming', icon: <InboxOutlined />, label: '来料录入' },
    { key: '/shipping', icon: <ShopOutlined />, label: '出货管理' }
  ]
}

export default function MainLayout() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  const items = user ? menuItems[user.role] || menuItems['admin'] : []

  const userMenu = [
    {
      key: '1',
      icon: <UserOutlined />,
      label: '个人信息',
      disabled: true
    },
    {
      type: 'divider'
    },
    {
      key: '2',
      icon: <LogoutOutlined />,
      label: '退出登录',
      onClick: handleLogout
    }
  ]

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider theme="dark" width={240}>
        <div style={{ 
          height: 64, 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          color: '#fff',
          fontSize: '18px',
          fontWeight: 'bold'
        }}>
          LCM MES
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[location.pathname]}
          items={items}
          onClick={({ key }) => navigate(key)}
        />
      </Sider>
      <Layout>
        <Header style={{ 
          background: '#fff', 
          padding: '0 24px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          boxShadow: '0 1px 4px rgba(0,0,0,0.1)'
        }}>
          <div style={{ fontSize: '18px', fontWeight: 'bold' }}>
            {user?.department || 'LCM MES系统'}
          </div>
          <Dropdown menu={{ items: userMenu }}>
            <Space style={{ cursor: 'pointer' }}>
              <Badge count={3} size="small">
                <Avatar icon={<UserOutlined />} />
              </Badge>
              <span>{user?.name || user?.username}</span>
            </Space>
          </Dropdown>
        </Header>
        <Content style={{ margin: '24px', background: '#fff', padding: 24, minHeight: 280 }}>
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  )
}
