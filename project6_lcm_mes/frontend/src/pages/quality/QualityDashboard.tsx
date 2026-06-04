import { useState, useEffect } from 'react'
import { Row, Col, Card, Statistic, Table, Tag, Alert, Button, Modal, Form, Select, Input, message } from 'antd'
import { CheckCircleOutlined, WarningOutlined, ExclamationCircleOutlined, PlusOutlined } from '@ant-design/icons'
import ReactECharts from 'echarts-for-react'

// 模拟数据
const mockDefects = [
  { id: 1, product_code: 'LCM-IP5-001-A001', stage: 1, defect_type: 'scratch', severity: 'major', status: 'pending' },
  { id: 2, product_code: 'LCM-IP5-002-B001', stage: 2, defect_type: 'spot', severity: 'minor', status: 'repairing' },
  { id: 3, product_code: 'LCM-IP5-003-C001', stage: 3, defect_type: 'crack', severity: 'critical', status: 'scrapped' },
  { id: 4, product_code: 'LCM-IP5-004-D001', stage: 4, defect_type: 'misalignment', severity: 'minor', status: 'repaired' },
  { id: 5, product_code: 'LCM-IP5-005-A002', stage: 1, defect_type: 'scratch', severity: 'major', status: 'pending' },
]

const mockInspections = [
  { id: 1, product_code: 'LCM-IP5-001-A001', type: 'visual', stage: 1, result: 'pass', time: '2024-05-28 09:30:00' },
  { id: 2, product_code: 'LCM-IP5-002-B001', type: 'electrical', stage: 2, result: 'fail', time: '2024-05-28 10:15:00' },
]

export default function QualityDashboard() {
  const [defects, setDefects] = useState<any[]>(mockDefects)
  const [inspections, setInspections] = useState<any[]>(mockInspections)
  const [loading, setLoading] = useState(false)
  const [inspectModalVisible, setInspectModalVisible] = useState(false)
  const [form] = Form.useForm()

  const fetchData = () => {
    // 使用模拟数据，无需调用API
    setLoading(false)
  }

  const spcChartOption = {
    title: { text: 'SPC控制图', left: 'center' },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: Array.from({ length: 20 }, (_, i) => i + 1) },
    yAxis: { type: 'value', min: 95, max: 100 },
    series: [
      {
        name: '实测值',
        type: 'line',
        data: [97.2, 97.5, 97.0, 97.3, 96.8, 97.4, 97.1, 96.9, 97.2, 97.0, 97.5, 97.2, 96.8, 97.3, 97.1, 97.4, 97.0, 97.2, 96.9, 97.3],
        itemStyle: { color: '#1890ff' }
      },
      {
        name: 'UCL',
        type: 'line',
        data: Array(20).fill(98.5),
        itemStyle: { color: '#ff4d4f' },
        lineStyle: { type: 'dashed' }
      },
      {
        name: 'LCL',
        type: 'line',
        data: Array(20).fill(95.5),
        itemStyle: { color: '#ff4d4f' },
        lineStyle: { type: 'dashed' }
      },
      {
        name: 'CL',
        type: 'line',
        data: Array(20).fill(97.0),
        itemStyle: { color: '#52c41a' },
        lineStyle: { type: 'solid' }
      }
    ]
  }

  const defectChartOption = {
    title: { text: '缺陷分布', left: 'center' },
    tooltip: { trigger: 'item' },
    legend: { bottom: 0 },
    series: [{
      type: 'pie',
      radius: '50%',
      data: [
        { value: 35, name: '划伤', itemStyle: { color: '#ff4d4f' } },
        { value: 25, name: '斑点', itemStyle: { color: '#faad14' } },
        { value: 20, name: '裂纹', itemStyle: { color: '#ff7875' } },
        { value: 12, name: '偏移', itemStyle: { color: '#1890ff' } },
        { value: 8, name: '其他', itemStyle: { color: '#8c8c8c' } }
      ]
    }]
  }

  const defectColumns = [
    { title: '产品编码', dataIndex: 'product_code', key: 'product_code' },
    { title: '阶段', dataIndex: 'stage', key: 'stage', render: (stage: number) => {
      const stages = ['大板切割', 'COG/FOG', '贴合', '背光模组']
      return <Tag color="blue">{stages[stage - 1]}</Tag>
    }},
    { title: '缺陷类型', dataIndex: 'defect_type', key: 'defect_type', render: (type: string) => {
      const texts: Record<string, string> = { scratch: '划伤', spot: '斑点', crack: '裂纹', misalignment: '偏移' }
      return texts[type] || type
    }},
    { title: '严重程度', dataIndex: 'severity', key: 'severity', render: (s: string) => {
      const colors: Record<string, string> = { critical: 'red', major: 'orange', minor: 'blue' }
      const texts: Record<string, string> = { critical: '严重', major: '主要', minor: '轻微' }
      return <Tag color={colors[s]}>{texts[s]}</Tag>
    }},
    { title: '状态', dataIndex: 'status', key: 'status', render: (s: string) => {
      const colors: Record<string, string> = { pending: 'orange', repairing: 'processing', repaired: 'green', scrapped: 'red' }
      const texts: Record<string, string> = { pending: '待处理', repairing: '返修中', repaired: '已修复', scrapped: '报废' }
      return <Tag color={colors[s]}>{texts[s]}</Tag>
    }},
  ]

  const handleSubmit = async (values: any) => {
    try {
      message.success('检测记录创建成功')
      setInspectModalVisible(false)
      fetchData()
    } catch (err) {
      message.error('创建失败')
    }
  }

  return (
    <div>
      <Alert message="质量监控中心" type="info" showIcon style={{ marginBottom: 16 }} />
      
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="今日检测数"
              value={3840}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="良率"
              value={97.2}
              suffix="%"
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="不良品数"
              value={108}
              prefix={<WarningOutlined />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="异常预警"
              value={2}
              prefix={<ExclamationCircleOutlined />}
              valueStyle={{ color: '#cf1322' }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={12}>
          <Card title="SPC控制图">
            <ReactECharts option={spcChartOption} style={{ height: 300 }} />
          </Card>
        </Col>
        <Col span={12}>
          <Card title="缺陷分布">
            <ReactECharts option={defectChartOption} style={{ height: 300 }} />
          </Card>
        </Col>
      </Row>

      <Card 
        title="不良品记录" 
        extra={<Button type="primary" icon={<PlusOutlined />} onClick={() => setInspectModalVisible(true)}>记录检测</Button>}
      >
        <Table 
          columns={defectColumns} 
          dataSource={defects}
          loading={loading}
          rowKey="id"
        />
      </Card>

      <Modal title="质量检测记录" open={inspectModalVisible} onCancel={() => setInspectModalVisible(false)} onOk={() => form.submit()}>
        <Form form={form} onFinish={handleSubmit} layout="vertical">
          <Form.Item name="product_code" label="产品编码" rules={[{ required: true }]}>
            <Input placeholder="输入产品编码" />
          </Form.Item>
          <Form.Item name="type" label="检测类型" rules={[{ required: true }]}>
            <Select>
              <Select.Option value="visual">外观检测</Select.Option>
              <Select.Option value="electrical">电性能检测</Select.Option>
              <Select.Option value="optical">光学检测</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="stage" label="阶段" rules={[{ required: true }]}>
            <Select>
              <Select.Option value={1}>大板切割</Select.Option>
              <Select.Option value={2}>COG/FOG</Select.Option>
              <Select.Option value={3}>贴合</Select.Option>
              <Select.Option value={4}>背光模组</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="result" label="检测结果" rules={[{ required: true }]}>
            <Select>
              <Select.Option value="pass">合格</Select.Option>
              <Select.Option value="fail">不合格</Select.Option>
              <Select.Option value="rework">需重工</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="note" label="备注">
            <Input.TextArea rows={4} />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}
