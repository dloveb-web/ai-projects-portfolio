import { useState, useEffect } from 'react'
import { Row, Col, Card, Statistic, Tag, Progress, Badge } from 'antd'
import { 
  ThunderboltOutlined, 
  CheckCircleOutlined, 
  WarningOutlined,
  ClockCircleOutlined,
  PackageOutlined,
  DesktopOutlined,
  UsersOutlined,
  DashboardOutlined,
  BarChartOutlined,
  PieChartOutlined,
  LineChartOutlined,
  LoadingOutlined
} from '@ant-design/icons'
import ReactECharts from 'echarts-for-react'

// 模拟数据类型
interface MockDashboardData {
  defect_trends: {
    trend_data: Array<{ hour: string; defect_count: number; defect_rate: number }>
    top_defects: Array<{ type: string; count: number; rate: number }>
  }
  quality_report: {
    total_units: number
    yield_rate: number
    defect_units: number
    defect_distribution: Array<{ type: string; count: number; percentage: number }>
    recommendations: Array<string>
    ai_summary: string
  }
  equipment_alerts: Array<{
    equipment_id: number
    equipment_name: string
    health_score: number
    risk_level: string
  }>
}

export default function DashboardScreen() {
  const [dashboardData, setDashboardData] = useState<MockDashboardData | null>(null)
  const [loading, setLoading] = useState(true)
  const [currentTime, setCurrentTime] = useState(new Date())

  // 生成模拟数据
  const generateMockData = () => {
    const now = new Date()
    const trend_data = []
    for (let i = 11; i >= 0; i--) {
      const hour = new Date(now.getTime() - i * 3600000)
      const baseDefectRate = 0.015 + Math.sin(i * 0.5) * 0.01
      trend_data.push({
        hour: hour.getHours().toString().padStart(2, '0') + ':00',
        defect_count: Math.floor(20 + Math.sin(i * 0.8) * 15),
        defect_rate: baseDefectRate + (Math.random() - 0.5) * 0.005
      })
    }

    return {
      defect_trends: {
        trend_data,
        top_defects: [
          { type: 'Mura缺陷', count: 45, rate: 0.028 },
          { type: '坏点', count: 32, rate: 0.020 },
          { type: '线缺陷', count: 25, rate: 0.016 }
        ]
      },
      quality_report: {
        total_units: 10850,
        yield_rate: 0.973,
        defect_units: 292,
        defect_distribution: [
          { type: 'Mura缺陷', count: 125, percentage: 42.8 },
          { type: '坏点', count: 85, percentage: 29.1 },
          { type: '线缺陷', count: 52, percentage: 17.8 },
          { type: '色不均', count: 20, percentage: 6.8 },
          { type: '其他', count: 10, percentage: 3.5 }
        ],
        recommendations: [
          '关注Mura缺陷上升趋势，建议检查偏光片贴合工艺',
          '坏点缺陷率维持稳定，继续保持当前管控水平',
          '建议增加COG工位的AOI检测频率至每小时一次',
          '背光模组批次质量良好，可继续使用当前供应商'
        ],
        ai_summary: 'AI分析显示当前综合良率97.3%处于正常水平，但Mura缺陷在近4小时有上升趋势，建议重点关注前段工艺，特别是大板切割后的清洗工序。COG绑定设备的健康度略降，建议安排预防性维护。'
      },
      equipment_alerts: [
        { equipment_id: 1, equipment_name: '大板切割机 #1', health_score: 0.95, risk_level: '低' },
        { equipment_id: 2, equipment_name: 'COG绑定机 #1', health_score: 0.65, risk_level: '高' },
        { equipment_id: 3, equipment_name: 'FOG绑定机 #1', health_score: 0.72, risk_level: '中' },
        { equipment_id: 4, equipment_name: '玻璃贴合机 #1', health_score: 0.88, risk_level: '低' }
      ]
    }
  }

  useEffect(() => {
    // 初始化数据
    setDashboardData(generateMockData())
    setLoading(false)

    const timer = setInterval(() => {
      setCurrentTime(new Date())
    }, 1000)

    // 每30秒更新一次模拟数据，添加一些波动
    const dataTimer = setInterval(() => {
      setDashboardData(prev => {
        if (!prev) return generateMockData()
        return {
          ...prev,
          quality_report: {
            ...prev.quality_report,
            total_units: prev.quality_report.total_units + Math.floor(Math.random() * 5),
            yield_rate: 0.97 + (Math.random() - 0.5) * 0.008
          }
        }
      })
    }, 30000)

    return () => {
      clearInterval(timer)
      clearInterval(dataTimer)
    }
  }, [])

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  }

  const formatDate = (date: Date) => {
    return date.toLocaleDateString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit', weekday: 'long' })
  }

  const productionTrendOption = {
    title: { 
      text: '实时产量趋势', 
      left: 'center',
      textStyle: { color: '#fff', fontSize: 16 }
    },
    tooltip: { trigger: 'axis' },
    legend: { data: ['产量', '目标'], bottom: 0, textStyle: { color: '#fff' } },
    grid: { left: '3%', right: '4%', bottom: '15%', top: '15%', containLabel: true },
    xAxis: {
      type: 'category',
      data: dashboardData?.defect_trends.trend_data.map(d => d.hour) || [],
      axisLabel: { color: '#999' }
    },
    yAxis: { 
      type: 'value', 
      axisLabel: { color: '#999' },
      min: 800,
      max: 1200
    },
    series: [
      { 
        name: '产量', 
        type: 'line', 
        smooth: true,
        data: dashboardData?.defect_trends.trend_data.map((_, idx) => 900 + Math.sin(idx * 0.5) * 150 + Math.floor(Math.random() * 100)) || [],
        itemStyle: { color: '#52c41a' },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(82, 196, 26, 0.3)' },
              { offset: 1, color: 'rgba(82, 196, 26, 0.05)' }
            ]
          }
        }
      },
      { 
        name: '目标', 
        type: 'line', 
        data: Array(12).fill(1000),
        itemStyle: { color: '#1890ff' },
        lineStyle: { type: 'dashed' }
      }
    ]
  }

  const defectRateOption = {
    title: { 
      text: '缺陷率趋势', 
      left: 'center',
      textStyle: { color: '#fff', fontSize: 16 }
    },
    tooltip: { trigger: 'axis' },
    legend: { data: ['缺陷率'], bottom: 0, textStyle: { color: '#fff' } },
    grid: { left: '3%', right: '4%', bottom: '15%', top: '15%', containLabel: true },
    xAxis: {
      type: 'category',
      data: dashboardData?.defect_trends.trend_data.map(d => d.hour) || [],
      axisLabel: { color: '#999' }
    },
    yAxis: { type: 'value', min: 0, max: 5, axisLabel: { color: '#999', formatter: '{value}%' } },
    series: [{
      name: '缺陷率',
      type: 'line',
      smooth: true,
      data: dashboardData?.defect_trends.trend_data.map(d => d.defect_rate * 100) || [],
      itemStyle: { color: '#f5222d' },
      areaStyle: {
        color: {
          type: 'linear',
          x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [
            { offset: 0, color: 'rgba(245, 34, 45, 0.3)' },
            { offset: 1, color: 'rgba(245, 34, 45, 0.05)' }
          ]
        }
      }
    }]
  }

  const defectDistributionOption = {
    title: { 
      text: '缺陷分布', 
      left: 'center',
      textStyle: { color: '#fff', fontSize: 16 }
    },
    tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
    legend: { bottom: 0, textStyle: { color: '#fff' } },
    series: [{
      type: 'pie',
      radius: ['40%', '70%'],
      center: ['50%', '50%'],
      avoidLabelOverlap: false,
      itemStyle: {
        borderRadius: 10,
        borderColor: '#1f1f1f',
        borderWidth: 2
      },
      label: { show: true, color: '#fff' },
      data: dashboardData?.quality_report.defect_distribution.map(d => ({
        value: d.count,
        name: d.type,
        itemStyle: {
          color: ['#f5222d', '#fa541c', '#faad14', '#52c41a', '#1890ff'][dashboardData?.quality_report.defect_distribution.indexOf(d) || 0]
        }
      })) || []
    }]
  }

  const equipmentStatusOption = {
    title: { 
      text: '设备运行状态', 
      left: 'center',
      textStyle: { color: '#fff', fontSize: 16 }
    },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    legend: { data: ['运行中', '待机', '维护'], bottom: 0, textStyle: { color: '#fff' } },
    grid: { left: '3%', right: '4%', bottom: '15%', top: '15%', containLabel: true },
    xAxis: {
      type: 'category',
      data: ['切割设备', 'COG设备', 'FOG设备', '贴合设备', '背光设备'],
      axisLabel: { color: '#999' }
    },
    yAxis: { type: 'value', axisLabel: { color: '#999' } },
    series: [
      { 
        name: '运行中', 
        type: 'bar', 
        data: [12, 8, 10, 15, 11],
        itemStyle: { color: '#52c41a' },
        stack: 'total'
      },
      { 
        name: '待机', 
        type: 'bar', 
        data: [2, 3, 2, 1, 2],
        itemStyle: { color: '#1890ff' },
        stack: 'total'
      },
      { 
        name: '维护', 
        type: 'bar', 
        data: [1, 1, 1, 0, 1],
        itemStyle: { color: '#faad14' },
        stack: 'total'
      }
    ]
  }

  const yieldTrendOption = {
    title: { 
      text: '良率趋势', 
      left: 'center',
      textStyle: { color: '#fff', fontSize: 16 }
    },
    tooltip: { trigger: 'axis' },
    legend: { data: ['良率', '目标'], bottom: 0, textStyle: { color: '#fff' } },
    grid: { left: '3%', right: '4%', bottom: '15%', top: '15%', containLabel: true },
    xAxis: {
      type: 'category',
      data: dashboardData?.defect_trends.trend_data.map(d => d.hour) || [],
      axisLabel: { color: '#999' }
    },
    yAxis: { type: 'value', min: 95, max: 99, axisLabel: { color: '#999', formatter: '{value}%' } },
    series: [
      {
        name: '良率',
        type: 'line',
        smooth: true,
        data: dashboardData?.defect_trends.trend_data.map((_, idx) => 96.5 + Math.sin(idx * 0.6) * 0.8 + Math.random() * 0.3) || [],
        itemStyle: { color: '#52c41a' },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(82, 196, 26, 0.3)' },
              { offset: 1, color: 'rgba(82, 196, 26, 0.05)' }
            ]
          }
        }
      },
      {
        name: '目标',
        type: 'line',
        data: dashboardData?.defect_trends.trend_data.map(() => 97) || [],
        itemStyle: { color: '#1890ff' },
        lineStyle: { type: 'dashed' }
      }
    ]
  }

  if (loading) {
    return (
      <div style={{ 
        height: '100vh', 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center',
        background: '#0a0a0f'
      }}>
        <LoadingOutlined style={{ fontSize: 48, color: '#1890ff', animation: 'spin 2s linear infinite' }} />
      </div>
    )
  }

  return (
    <div style={{ 
      minHeight: '100vh', 
      background: '#0a0a0f', 
      padding: '20px',
      color: '#fff'
    }}>
      <Row gutter={16} style={{ marginBottom: 20 }}>
        <Col span={18}>
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: 16,
            fontSize: 28,
            fontWeight: 'bold',
            color: '#1890ff'
          }}>
            <DashboardOutlined />
            <span>LCM MES 智能监控大屏</span>
          </div>
        </Col>
        <Col span={6} style={{ textAlign: 'right' }}>
          <div style={{ fontSize: 24, fontWeight: 'bold', color: '#fff' }}>
            {formatTime(currentTime)}
          </div>
          <div style={{ fontSize: 14, color: '#999' }}>
            {formatDate(currentTime)}
          </div>
        </Col>
      </Row>

      <Row gutter={16} style={{ marginBottom: 20 }}>
        <Col span={4}>
          <Card 
            style={{ background: 'linear-gradient(135deg, #1f1f2e 0%, #2a2a3e 100%)', borderRadius: 12 }}
            styles={{ body: { padding: 20 } }}
          >
            <Statistic
              title="今日产出"
              value={dashboardData?.quality_report.total_units || 0}
              prefix={<ThunderboltOutlined style={{ color: '#52c41a' }} />}
              valueStyle={{ color: '#fff', fontSize: 32 }}
              titleStyle={{ color: '#999' }}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card 
            style={{ background: 'linear-gradient(135deg, #1f1f2e 0%, #2a2a3e 100%)', borderRadius: 12 }}
            styles={{ body: { padding: 20 } }}
          >
            <Statistic
              title="综合良率"
              value={(dashboardData?.quality_report.yield_rate || 0) * 100}
              suffix="%"
              prefix={<CheckCircleOutlined style={{ color: '#52c41a' }} />}
              valueStyle={{ color: '#fff', fontSize: 32 }}
              titleStyle={{ color: '#999' }}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card 
            style={{ background: 'linear-gradient(135deg, #1f1f2e 0%, #2a2a3e 100%)', borderRadius: 12 }}
            styles={{ body: { padding: 20 } }}
          >
            <Statistic
              title="不良品数"
              value={dashboardData?.quality_report.defect_units || 0}
              prefix={<WarningOutlined style={{ color: '#f5222d' }} />}
              valueStyle={{ color: '#fff', fontSize: 32 }}
              titleStyle={{ color: '#999' }}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card 
            style={{ background: 'linear-gradient(135deg, #1f1f2e 0%, #2a2a3e 100%)', borderRadius: 12 }}
            styles={{ body: { padding: 20 } }}
          >
            <Statistic
              title="设备预警"
              value={dashboardData?.equipment_alerts.filter(e => e.risk_level === '高').length || 0}
              prefix={<DesktopOutlined style={{ color: '#faad14' }} />}
              valueStyle={{ color: '#fff', fontSize: 32 }}
              titleStyle={{ color: '#999' }}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card 
            style={{ background: 'linear-gradient(135deg, #1f1f2e 0%, #2a2a3e 100%)', borderRadius: 12 }}
            styles={{ body: { padding: 20 } }}
          >
            <Statistic
              title="生产效率"
              value={94.5}
              suffix="%"
              prefix={<BarChartOutlined style={{ color: '#1890ff' }} />}
              valueStyle={{ color: '#fff', fontSize: 32 }}
              titleStyle={{ color: '#999' }}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card 
            style={{ background: 'linear-gradient(135deg, #1f1f2e 0%, #2a2a3e 100%)', borderRadius: 12 }}
            styles={{ body: { padding: 20 } }}
          >
            <Statistic
              title="在产工单"
              value={12}
              prefix={<ClockCircleOutlined style={{ color: '#722ed1' }} />}
              valueStyle={{ color: '#fff', fontSize: 32 }}
              titleStyle={{ color: '#999' }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={16} style={{ marginBottom: 20 }}>
        <Col span={8}>
          <Card 
            style={{ background: '#12121a', borderRadius: 12, border: 'none' }}
            styles={{ body: { padding: 16 } }}
          >
            <ReactECharts option={productionTrendOption} style={{ height: 280 }} />
          </Card>
        </Col>
        <Col span={8}>
          <Card 
            style={{ background: '#12121a', borderRadius: 12, border: 'none' }}
            styles={{ body: { padding: 16 } }}
          >
            <ReactECharts option={yieldTrendOption} style={{ height: 280 }} />
          </Card>
        </Col>
        <Col span={8}>
          <Card 
            style={{ background: '#12121a', borderRadius: 12, border: 'none' }}
            styles={{ body: { padding: 16 } }}
          >
            <ReactECharts option={defectRateOption} style={{ height: 280 }} />
          </Card>
        </Col>
      </Row>

      <Row gutter={16} style={{ marginBottom: 20 }}>
        <Col span={8}>
          <Card 
            style={{ background: '#12121a', borderRadius: 12, border: 'none' }}
            styles={{ body: { padding: 16 } }}
          >
            <ReactECharts option={defectDistributionOption} style={{ height: 280 }} />
          </Card>
        </Col>
        <Col span={8}>
          <Card 
            style={{ background: '#12121a', borderRadius: 12, border: 'none' }}
            styles={{ body: { padding: 16 } }}
          >
            <ReactECharts option={equipmentStatusOption} style={{ height: 280 }} />
          </Card>
        </Col>
        <Col span={8}>
          <Card 
            style={{ background: '#12121a', borderRadius: 12, border: 'none' }}
            styles={{ body: { padding: 16 } }}
          >
            <div style={{ fontSize: 16, fontWeight: 'bold', marginBottom: 16, color: '#fff' }}>
              <PieChartOutlined /> 设备健康状态
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {dashboardData?.equipment_alerts.map((eq, idx) => (
                <div key={eq.equipment_id} style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: 12,
                  padding: 12,
                  background: '#1a1a24',
                  borderRadius: 8
                }}>
                  <div style={{ width: 40, height: 40, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: eq.risk_level === '高' ? '#f5222d22' : eq.risk_level === '中' ? '#faad1422' : '#52c41a22' }}>
                    <DesktopOutlined style={{ color: eq.risk_level === '高' ? '#f5222d' : eq.risk_level === '中' ? '#faad14' : '#52c41a' }} />
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ color: '#fff', fontWeight: 'bold' }}>{eq.equipment_name}</div>
                    <div style={{ color: '#999', fontSize: 12 }}>健康度: {(eq.health_score * 100).toFixed(1)}%</div>
                  </div>
                  <Tag color={eq.risk_level === '高' ? 'red' : eq.risk_level === '中' ? 'orange' : 'green'}>
                    {eq.risk_level}风险
                  </Tag>
                </div>
              ))}
            </div>
          </Card>
        </Col>
      </Row>

      <Row gutter={16}>
        <Col span={12}>
          <Card 
            style={{ background: '#12121a', borderRadius: 12, border: 'none' }}
            styles={{ body: { padding: 16 } }}
          >
            <div style={{ fontSize: 16, fontWeight: 'bold', marginBottom: 16, color: '#fff' }}>
              <BarChartOutlined /> TOP缺陷排行
            </div>
            <div style={{ display: 'flex', gap: 16 }}>
              {dashboardData?.defect_trends.top_defects.map((defect, idx) => (
                <div key={defect.type} style={{ flex: 1 }}>
                  <div style={{ 
                    height: 120, 
                    background: `linear-gradient(180deg, #${(idx === 0 ? 'f5222d' : idx === 1 ? 'fa541c' : 'faad14')} 0%, #${(idx === 0 ? 'f5222d44' : idx === 1 ? 'fa541c44' : 'faad1444')} 100%)`,
                    borderRadius: 8,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    position: 'relative'
                  }}>
                    <div style={{ fontSize: 32, fontWeight: 'bold', color: '#fff' }}>{defect.count}</div>
                    <div style={{ fontSize: 12, color: '#fff9' }}>{(defect.rate * 100).toFixed(2)}%</div>
                  </div>
                  <div style={{ textAlign: 'center', marginTop: 8, color: '#999', fontSize: 12 }}>
                    {defect.type}
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </Col>
        <Col span={12}>
          <Card 
            style={{ background: '#12121a', borderRadius: 12, border: 'none' }}
            styles={{ body: { padding: 16 } }}
          >
            <div style={{ fontSize: 16, fontWeight: 'bold', marginBottom: 16, color: '#fff' }}>
              <LineChartOutlined /> AI智能建议
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {dashboardData?.quality_report.recommendations.map((rec, idx) => (
                <div key={idx} style={{ 
                  display: 'flex', 
                  alignItems: 'flex-start', 
                  gap: 12,
                  padding: 12,
                  background: '#1a1a24',
                  borderRadius: 8,
                  borderLeft: '4px solid #1890ff'
                }}>
                  <CheckCircleOutlined style={{ color: '#52c41a', marginTop: 2 }} />
                  <span style={{ color: '#ccc' }}>{rec}</span>
                </div>
              ))}
            </div>
            <div style={{ marginTop: 16, padding: 12, background: '#1a1a24', borderRadius: 8 }}>
              <div style={{ color: '#1890ff', marginBottom: 4 }}>AI分析摘要</div>
              <div style={{ color: '#ccc', fontSize: 12 }}>
                {dashboardData?.quality_report.ai_summary}
              </div>
            </div>
          </Card>
        </Col>
      </Row>
    </div>
  )
}