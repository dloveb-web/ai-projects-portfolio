import { useState, useEffect } from 'react'
import { Row, Col, Card, Statistic, Table, Progress, Tag, Alert } from 'antd'
import { 
  ThunderboltOutlined, 
  CheckCircleOutlined, 
  WarningOutlined,
  ClockCircleOutlined
} from '@ant-design/icons'
import ReactECharts from 'echarts-for-react'
import { productionApi } from '../../api/production'
import { equipmentApi } from '../../api/equipment'

export default function EngineeringDashboard() {
  const [dashboardData, setDashboardData] = useState<any>(null)
  const [equipmentList, setEquipmentList] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    try {
      const [dashRes, eqRes] = await Promise.all([
        productionApi.getDashboard(),
        equipmentApi.getEquipment()
      ])
      setDashboardData(dashRes.data)
      setEquipmentList(eqRes.data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const productionChartOption = {
    title: { text: '四阶段产出统计', left: 'center' },
    tooltip: { trigger: 'axis' },
    legend: { data: ['计划', '实际'], bottom: 0 },
    xAxis: {
      type: 'category',
      data: ['大板切割', 'COG/FOG', '贴合', '背光模组']
    },
    yAxis: { type: 'value' },
    series: [
      { name: '计划', type: 'bar', data: [1000, 950, 920, 900], itemStyle: { color: '#1890ff' } },
      { name: '实际', type: 'bar', data: [980, 940, 910, 890], itemStyle: { color: '#52c41a' } }
    ]
  }

  const yieldChartOption = {
    title: { text: '良率趋势', left: 'center' },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: ['0:00', '4:00', '8:00', '12:00', '16:00', '20:00'] },
    yAxis: { type: 'value', min: 90, max: 100 },
    series: [{
      name: '良率',
      type: 'line',
      smooth: true,
      data: [96.5, 97.2, 96.8, 97.5, 97.0, 97.3],
      itemStyle: { color: '#52c41a' }
    }]
  }

  const equipmentColumns = [
    { title: '设备编号', dataIndex: 'code', key: 'code' },
    { title: '设备名称', dataIndex: 'name', key: 'name' },
    { title: '状态', dataIndex: 'status', key: 'status', render: (status: string) => {
      const colors: Record<string, string> = { running: 'green', idle: 'blue', maintenance: 'orange', fault: 'red' }
      const texts: Record<string, string> = { running: '运行', idle: '待机', maintenance: '维护', fault: '故障' }
      return <Tag color={colors[status]}>{texts[status]}</Tag>
    }},
    { title: '健康度', dataIndex: 'health_score', key: 'health_score', render: (score: number) => (
      <Progress percent={score} size="small" status={score > 90 ? 'normal' : score > 70 ? 'exception' : 'exception'} />
    )}
  ]

  return (
    <div>
      <Alert message="全局监控仪表盘" type="info" showIcon style={{ marginBottom: 16 }} />
      
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="今日产出"
              value={dashboardData?.today_output || 3840}
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="综合良率"
              value={dashboardData?.yield_rate || 97.2}
              suffix="%"
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="在产工单"
              value={dashboardData?.active_orders || 12}
              prefix={<ClockCircleOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="设备预警"
              value={dashboardData?.equipment_alerts || 2}
              prefix={<WarningOutlined />}
              valueStyle={{ color: '#cf1322' }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={12}>
          <Card title="生产统计">
            <ReactECharts option={productionChartOption} style={{ height: 300 }} />
          </Card>
        </Col>
        <Col span={12}>
          <Card title="良率趋势">
            <ReactECharts option={yieldChartOption} style={{ height: 300 }} />
          </Card>
        </Col>
      </Row>

      <Card title="设备状态监控">
        <Table 
          columns={equipmentColumns} 
          dataSource={equipmentList}
          loading={loading}
          rowKey="id"
        />
      </Card>
    </div>
  )
}
