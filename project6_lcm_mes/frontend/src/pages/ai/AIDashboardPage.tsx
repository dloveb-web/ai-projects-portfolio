import { useState, useEffect } from 'react'
import { 
  Row, Col, Card, Statistic, Table, Tag, Button, 
  Modal, Form, Input, Select, Progress, Empty, Spin, Space
} from 'antd'
import { 
  RocketOutlined, WarningOutlined, 
  BarChartOutlined, CheckCircleOutlined,
  DashboardOutlined, DesktopOutlined, ShopOutlined
} from '@ant-design/icons'
import ReactECharts from 'echarts-for-react'
import { aiApi, 
  DefectPredictionRequest, 
  EquipmentHealthRequest, 
  RepairDecisionRequest,
  YieldPredictionRequest,
  AIDashboardResponse,
  DefectPredictionResponse,
  EquipmentHealthResponse,
  RepairDecisionResponse,
  YieldPredictionResponse
} from '../../api/ai'

const { Option } = Select

export default function AIDashboardPage() {
  const [dashboardData, setDashboardData] = useState<AIDashboardResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState<'dashboard' | 'defect' | 'equipment' | 'repair' | 'yield'>('dashboard')
  
  const [defectModalVisible, setDefectModalVisible] = useState(false)
  const [equipmentModalVisible, setEquipmentModalVisible] = useState(false)
  const [repairModalVisible, setRepairModalVisible] = useState(false)
  const [yieldModalVisible, setYieldModalVisible] = useState(false)
  
  const [defectForm] = Form.useForm()
  const [equipmentForm] = Form.useForm()
  const [repairForm] = Form.useForm()
  const [yieldForm] = Form.useForm()
  
  const [defectResult, setDefectResult] = useState<DefectPredictionResponse | null>(null)
  const [equipmentResult, setEquipmentResult] = useState<EquipmentHealthResponse | null>(null)
  const [repairResult, setRepairResult] = useState<RepairDecisionResponse | null>(null)
  const [yieldResult, setYieldResult] = useState<YieldPredictionResponse | null>(null)

  useEffect(() => {
    fetchDashboardData()
  }, [])

  const fetchDashboardData = async () => {
    try {
      const res = await aiApi.getAIDashboard()
      setDashboardData(res.data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handleDefectPrediction = async () => {
    try {
      const values = defectForm.getFieldsValue() as DefectPredictionRequest
      const res = await aiApi.predictDefect(values)
      setDefectResult(res.data)
    } catch (err) {
      console.error(err)
    }
  }

  const handleEquipmentHealth = async () => {
    try {
      const values = equipmentForm.getFieldsValue() as EquipmentHealthRequest
      const res = await aiApi.getEquipmentHealth(values)
      setEquipmentResult(res.data)
    } catch (err) {
      console.error(err)
    }
  }

  const handleRepairDecision = async () => {
    try {
      const values = repairForm.getFieldsValue() as RepairDecisionRequest
      const res = await aiApi.makeRepairDecision(values)
      setRepairResult(res.data)
    } catch (err) {
      console.error(err)
    }
  }

  const handleYieldPrediction = async () => {
    try {
      const values = yieldForm.getFieldsValue() as YieldPredictionRequest
      const res = await aiApi.predictYield(values)
      setYieldResult(res.data)
    } catch (err) {
      console.error(err)
    }
  }

  const defectTrendOption = {
    title: { text: '缺陷趋势', left: 'center' },
    tooltip: { trigger: 'axis' },
    legend: { data: ['缺陷数', '缺陷率'], bottom: 0 },
    xAxis: {
      type: 'category',
      data: dashboardData?.defect_trends.trend_data.map(d => d.hour) || []
    },
    yAxis: [
      { type: 'value', name: '缺陷数' },
      { type: 'value', name: '缺陷率(%)', axisLabel: { formatter: '{value}%' } }
    ],
    series: [
      {
        name: '缺陷数',
        type: 'bar',
        data: dashboardData?.defect_trends.trend_data.map(d => d.defect_count) || [],
        itemStyle: { color: '#f5222d' }
      },
      {
        name: '缺陷率',
        type: 'line',
        yAxisIndex: 1,
        data: dashboardData?.defect_trends.trend_data.map(d => d.defect_rate * 100) || [],
        smooth: true,
        itemStyle: { color: '#1890ff' }
      }
    ]
  }

  const defectDistributionOption = {
    title: { text: '缺陷分布', left: 'center' },
    tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
    legend: { bottom: 0 },
    series: [{
      type: 'pie',
      radius: ['40%', '70%'],
      avoidLabelOverlap: false,
      itemStyle: {
        borderRadius: 10,
        borderColor: '#fff',
        borderWidth: 2
      },
      label: { show: true },
      data: dashboardData?.quality_report.defect_distribution.map(d => ({
        value: d.count,
        name: d.type,
        itemStyle: {
          color: ['#f5222d', '#fa541c', '#faad14', '#52c41a', '#1890ff'][dashboardData?.quality_report.defect_distribution.indexOf(d) || 0]
        }
      })) || []
    }]
  }

  const topDefectColumns = [
    { title: '缺陷类型', dataIndex: 'type', key: 'type' },
    { title: '数量', dataIndex: 'count', key: 'count' },
    { title: '比率', dataIndex: 'rate', key: 'rate', render: (rate: number) => `${(rate * 100).toFixed(2)}%` }
  ]

  const equipmentAlertColumns = [
    { title: '设备名称', dataIndex: 'equipment_name', key: 'equipment_name' },
    { title: '健康评分', dataIndex: 'health_score', key: 'health_score', render: (score: number) => (
      <Progress percent={score * 100} size="small" status={score > 0.85 ? 'success' : score > 0.7 ? 'warning' : 'error'} />
    )},
    { title: '风险等级', dataIndex: 'risk_level', key: 'risk_level', render: (level: string) => (
      <Tag color={level === '高' ? 'red' : level === '中' ? 'orange' : 'green'}>{level}</Tag>
    )},
    { title: '预计故障', dataIndex: 'predicted_failure_date', key: 'predicted_failure_date' }
  ]

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '200px' }}>
        <Spin size="large" />
      </div>
    )
  }

  return (
    <div>
      <Card title={
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <RocketOutlined style={{ fontSize: 20 }} />
          <span>AI智能分析中心</span>
        </div>
      }>
        <div style={{ marginBottom: 16 }}>
          <Space.Compact>
            <Button 
              type={activeTab === 'dashboard' ? 'primary' : 'default'} 
              onClick={() => setActiveTab('dashboard')}
            >
              <DashboardOutlined /> 总览
            </Button>
            <Button 
              type={activeTab === 'defect' ? 'primary' : 'default'} 
              onClick={() => setActiveTab('defect')}
            >
              <WarningOutlined /> 缺陷预测
            </Button>
            <Button 
              type={activeTab === 'equipment' ? 'primary' : 'default'} 
              onClick={() => setActiveTab('equipment')}
            >
              <DesktopOutlined /> 设备健康
            </Button>
            <Button 
              type={activeTab === 'repair' ? 'primary' : 'default'} 
              onClick={() => setActiveTab('repair')}
            >
              <ShopOutlined /> 返修决策
            </Button>
            <Button 
              type={activeTab === 'yield' ? 'primary' : 'default'} 
              onClick={() => setActiveTab('yield')}
            >
              <BarChartOutlined /> 良率预测
            </Button>
          </Space.Compact>
        </div>

        {activeTab === 'dashboard' && (
          <div>
            <Row gutter={16} style={{ marginBottom: 24 }}>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="综合良率"
                    value={(dashboardData?.quality_report.yield_rate || 0) * 100}
                    suffix="%"
                    prefix={<CheckCircleOutlined />}
                    valueStyle={{ color: '#52c41a' }}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="缺陷率"
                    value={(dashboardData?.defect_trends.overall_defect_rate || 0) * 100}
                    suffix="%"
                    prefix={<WarningOutlined />}
                    valueStyle={{ color: '#f5222d' }}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="今日产量"
                    value={dashboardData?.quality_report.total_units || 0}
                    prefix={<BarChartOutlined />}
                    valueStyle={{ color: '#1890ff' }}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="本周预测良率"
                    value={(dashboardData?.yield_forecast.week.predicted_yield || 0) * 100}
                    suffix="%"
                    prefix={<BarChartOutlined />}
                    valueStyle={{ color: '#722ed1' }}
                  />
                </Card>
              </Col>
            </Row>

            <Row gutter={16} style={{ marginBottom: 24 }}>
              <Col span={12}>
                <Card title="缺陷趋势">
                  <ReactECharts option={defectTrendOption} style={{ height: 300 }} />
                </Card>
              </Col>
              <Col span={12}>
                <Card title="缺陷分布">
                  <ReactECharts option={defectDistributionOption} style={{ height: 300 }} />
                </Card>
              </Col>
            </Row>

            <Row gutter={16}>
              <Col span={12}>
                <Card title="TOP缺陷">
                  <Table 
                    columns={topDefectColumns}
                    dataSource={dashboardData?.defect_trends.top_defects}
                    rowKey="type"
                    pagination={false}
                  />
                </Card>
              </Col>
              <Col span={12}>
                <Card title="设备预警">
                  <Table 
                    columns={equipmentAlertColumns}
                    dataSource={dashboardData?.equipment_alerts}
                    rowKey="equipment_id"
                    pagination={false}
                  />
                </Card>
              </Col>
            </Row>
          </div>
        )}

        {activeTab === 'defect' && (
          <div>
            <Row gutter={16}>
              <Col span={12}>
                <Card title="缺陷预测参数">
                  <Form form={defectForm} layout="vertical">
                    <Form.Item name="product_code" label="产品编码" rules={[{ required: true }]}>
                      <Input placeholder="输入产品编码" />
                    </Form.Item>
                    <Form.Item name="process_step" label="工艺步骤" rules={[{ required: true }]}>
                      <Select placeholder="选择工艺步骤">
                        <Option value="cutting">大板切割</Option>
                        <Option value="cog">COG绑定</Option>
                        <Option value="fog">FOG绑定</Option>
                        <Option value="bonding">玻璃盖板贴合</Option>
                        <Option value="backlight">背光模组组装</Option>
                      </Select>
                    </Form.Item>
                    <Form.Item name="equipment_id" label="设备ID">
                      <Input type="number" placeholder="输入设备ID" />
                    </Form.Item>
                    <Form.Item name="operator_id" label="操作员ID">
                      <Input type="number" placeholder="输入操作员ID" />
                    </Form.Item>
                    <Button type="primary" onClick={handleDefectPrediction} block>
                      执行预测
                    </Button>
                  </Form>
                </Card>
              </Col>
              <Col span={12}>
                <Card title="预测结果">
                  {defectResult ? (
                    <div>
                      <div style={{ marginBottom: 16 }}>
                        <Statistic
                          title="缺陷概率"
                          value={defectResult.defect_probability * 100}
                          suffix="%"
                          valueStyle={{ color: defectResult.defect_probability > 0.2 ? '#f5222d' : '#52c41a' }}
                        />
                      </div>
                      {defectResult.defect_type && (
                        <Tag color="red" style={{ marginBottom: 16 }}>
                          预测缺陷类型: {defectResult.defect_type}
                        </Tag>
                      )}
                      <div style={{ marginBottom: 16 }}>
                        <span>置信度: {defectResult.confidence * 100}%</span>
                      </div>
                      <div>
                        <h4>建议措施:</h4>
                        <ul>
                          {defectResult.recommendations.map((rec, idx) => (
                            <li key={idx}>{rec}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  ) : (
                    <Empty description="请输入参数并执行预测" />
                  )}
                </Card>
              </Col>
            </Row>
          </div>
        )}

        {activeTab === 'equipment' && (
          <div>
            <Row gutter={16}>
              <Col span={12}>
                <Card title="设备健康查询">
                  <Form form={equipmentForm} layout="vertical">
                    <Form.Item name="equipment_id" label="设备ID" rules={[{ required: true }]}>
                      <Input type="number" placeholder="输入设备ID" />
                    </Form.Item>
                    <Form.Item name="look_ahead_days" label="预测天数">
                      <Select placeholder="选择预测天数">
                        <Option value={7}>7天</Option>
                        <Option value={14}>14天</Option>
                        <Option value={30}>30天</Option>
                      </Select>
                    </Form.Item>
                    <Button type="primary" onClick={handleEquipmentHealth} block>
                      查询健康状态
                    </Button>
                  </Form>
                </Card>
              </Col>
              <Col span={12}>
                <Card title="健康评估结果">
                  {equipmentResult ? (
                    <div>
                      <div style={{ marginBottom: 16 }}>
                        <span>设备名称: {equipmentResult.equipment_name}</span>
                      </div>
                      <div style={{ marginBottom: 16 }}>
                        <Statistic
                          title="健康评分"
                          value={equipmentResult.health_score * 100}
                          suffix="%"
                          valueStyle={{ 
                            color: equipmentResult.health_score > 0.85 ? '#52c41a' : 
                                   equipmentResult.health_score > 0.7 ? '#faad14' : '#f5222d' 
                          }}
                        />
                      </div>
                      <Tag color={equipmentResult.risk_level === '高' ? 'red' : 
                                  equipmentResult.risk_level === '中' ? 'orange' : 'green'} 
                           style={{ marginBottom: 16 }}>
                        风险等级: {equipmentResult.risk_level}
                      </Tag>
                      {equipmentResult.predicted_failure_date && (
                        <div style={{ marginBottom: 16 }}>
                          <span>预计故障日期: {equipmentResult.predicted_failure_date}</span>
                        </div>
                      )}
                      <div>
                        <h4>建议操作:</h4>
                        <ul>
                          {equipmentResult.recommended_actions.map((action, idx) => (
                            <li key={idx}>{action}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  ) : (
                    <Empty description="请输入设备ID并查询" />
                  )}
                </Card>
              </Col>
            </Row>
          </div>
        )}

        {activeTab === 'repair' && (
          <div>
            <Row gutter={16}>
              <Col span={12}>
                <Card title="返修决策参数">
                  <Form form={repairForm} layout="vertical">
                    <Form.Item name="product_code" label="产品编码" rules={[{ required: true }]}>
                      <Input placeholder="输入产品编码" />
                    </Form.Item>
                    <Form.Item name="defect_type" label="缺陷类型" rules={[{ required: true }]}>
                      <Select placeholder="选择缺陷类型">
                        <Option value="mura">Mura缺陷</Option>
                        <Option value="dead_pixel">坏点</Option>
                        <Option value="line_defect">线缺陷</Option>
                        <Option value="color_uniformity">色不均</Option>
                        <Option value="brightness_issue">亮度问题</Option>
                      </Select>
                    </Form.Item>
                    <Form.Item name="defect_severity" label="缺陷严重程度" rules={[{ required: true }]}>
                      <Select placeholder="选择严重程度">
                        <Option value="critical">严重</Option>
                        <Option value="major">主要</Option>
                        <Option value="minor">次要</Option>
                      </Select>
                    </Form.Item>
                    <Button type="primary" onClick={handleRepairDecision} block>
                      获取决策建议
                    </Button>
                  </Form>
                </Card>
              </Col>
              <Col span={12}>
                <Card title="决策结果">
                  {repairResult ? (
                    <div>
                      <div style={{ marginBottom: 16 }}>
                        <Tag color={repairResult.should_repair ? 'green' : 'red'}>
                          {repairResult.should_repair ? '建议返修' : '建议报废'}
                        </Tag>
                      </div>
                      <div style={{ marginBottom: 16 }}>
                        <span>置信度: {repairResult.confidence * 100}%</span>
                      </div>
                      {repairResult.should_repair && (
                        <div style={{ marginBottom: 16 }}>
                          <Statistic title="预计成本" value={repairResult.repair_cost_estimate} suffix="元" />
                          <Statistic title="预计时间" value={repairResult.repair_time_estimate} suffix="分钟" />
                        </div>
                      )}
                      {repairResult.alternative_action && (
                        <div style={{ marginBottom: 16 }}>
                          <span>替代方案: {repairResult.alternative_action}</span>
                        </div>
                      )}
                    </div>
                  ) : (
                    <Empty description="请输入参数并获取决策" />
                  )}
                </Card>
              </Col>
            </Row>
          </div>
        )}

        {activeTab === 'yield' && (
          <div>
            <Row gutter={16}>
              <Col span={12}>
                <Card title="良率预测参数">
                  <Form form={yieldForm} layout="vertical">
                    <Form.Item name="product_type" label="产品类型" rules={[{ required: true }]}>
                      <Select placeholder="选择产品类型">
                        <Option value="LCM">LCM模组</Option>
                        <Option value="COG">COG绑定</Option>
                        <Option value="FOG">FOG绑定</Option>
                      </Select>
                    </Form.Item>
                    <Form.Item name="batch_size" label="批次数量" rules={[{ required: true }]}>
                      <Input type="number" placeholder="输入批次数量" />
                    </Form.Item>
                    <Button type="primary" onClick={handleYieldPrediction} block>
                      预测良率
                    </Button>
                  </Form>
                </Card>
              </Col>
              <Col span={12}>
                <Card title="预测结果">
                  {yieldResult ? (
                    <div>
                      <div style={{ marginBottom: 16 }}>
                        <Statistic
                          title="预测良率"
                          value={yieldResult.predicted_yield * 100}
                          suffix="%"
                          valueStyle={{ color: '#52c41a' }}
                        />
                      </div>
                      <div style={{ marginBottom: 16 }}>
                        <span>置信度: {yieldResult.confidence * 100}%</span>
                      </div>
                      <div style={{ marginBottom: 16 }}>
                        <h4>关键影响因素:</h4>
                        <ul>
                          {yieldResult.key_factors.map((factor, idx) => (
                            <li key={idx}>{factor}</li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <h4>优化建议:</h4>
                        <ul>
                          {yieldResult.optimization_suggestions.map((suggestion, idx) => (
                            <li key={idx}>{suggestion}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  ) : (
                    <Empty description="请输入参数并执行预测" />
                  )}
                </Card>
              </Col>
            </Row>
          </div>
        )}
      </Card>

      <Modal
        title="缺陷预测"
        open={defectModalVisible}
        onCancel={() => setDefectModalVisible(false)}
        footer={null}
      >
      </Modal>

      <Modal
        title="设备健康"
        open={equipmentModalVisible}
        onCancel={() => setEquipmentModalVisible(false)}
        footer={null}
      >
      </Modal>

      <Modal
        title="返修决策"
        open={repairModalVisible}
        onCancel={() => setRepairModalVisible(false)}
        footer={null}
      >
      </Modal>

      <Modal
        title="良率预测"
        open={yieldModalVisible}
        onCancel={() => setYieldModalVisible(false)}
        footer={null}
      >
      </Modal>
    </div>
  )
}