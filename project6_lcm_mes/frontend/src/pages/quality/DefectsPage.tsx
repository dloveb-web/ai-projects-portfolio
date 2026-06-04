import { useState, useEffect } from 'react'
import { Card, Table, Tag, Button, Modal, Form, Input, Select, message } from 'antd'
import { SearchOutlined } from '@ant-design/icons'
import { qualityApi } from '../../api/quality'

export default function DefectsPage() {
  const [defects, setDefects] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [searchModalVisible, setSearchModalVisible] = useState(false)
  const [form] = Form.useForm()

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    try {
      const res = await qualityApi.getDefects()
      setDefects(res.data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const columns = [
    { title: '产品编码', dataIndex: 'product_code', key: 'product_code' },
    { title: '阶段', dataIndex: 'stage', key: 'stage', render: (stage: number) => {
      const stages = ['大板切割', 'COG/FOG', '贴合', '背光模组']
      return <Tag color="blue">{stages[stage - 1]}</Tag>
    }},
    { title: '缺陷类型', dataIndex: 'defect_type', key: 'defect_type', render: (type: string) => {
      const colors: Record<string, string> = { scratch: 'red', spot: 'orange', crack: 'red', misalignment: 'blue' }
      const texts: Record<string, string> = { scratch: '划伤', spot: '斑点', crack: '裂纹', misalignment: '偏移' }
      return <Tag color={colors[type]}>{texts[type] || type}</Tag>
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
    { title: '返修次数', dataIndex: 'repair_count', key: 'repair_count' },
    { title: '发现时间', dataIndex: 'created_at', key: 'created_at' },
  ]

  const handleSearch = async (values: any) => {
    message.success('搜索成功')
  }

  return (
    <div>
      <Card 
        title="不良品管理" 
        extra={<Button icon={<SearchOutlined />} onClick={() => setSearchModalVisible(true)}>搜索</Button>}
      >
        <Table 
          columns={columns} 
          dataSource={defects}
          loading={loading}
          rowKey="id"
        />
      </Card>

      <Modal title="搜索不良品" open={searchModalVisible} onCancel={() => setSearchModalVisible(false)} onOk={() => form.submit()}>
        <Form form={form} onFinish={handleSearch} layout="vertical">
          <Form.Item name="product_code" label="产品编码">
            <Input placeholder="输入产品编码" />
          </Form.Item>
          <Form.Item name="stage" label="阶段">
            <Select placeholder="选择阶段">
              <Select.Option value={1}>大板切割</Select.Option>
              <Select.Option value={2}>COG/FOG</Select.Option>
              <Select.Option value={3}>贴合</Select.Option>
              <Select.Option value={4}>背光模组</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="defect_type" label="缺陷类型">
            <Select placeholder="选择缺陷类型">
              <Select.Option value="scratch">划伤</Select.Option>
              <Select.Option value="spot">斑点</Select.Option>
              <Select.Option value="crack">裂纹</Select.Option>
              <Select.Option value="misalignment">偏移</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="status" label="状态">
            <Select placeholder="选择状态">
              <Select.Option value="pending">待处理</Select.Option>
              <Select.Option value="repairing">返修中</Select.Option>
              <Select.Option value="repaired">已修复</Select.Option>
              <Select.Option value="scrapped">报废</Select.Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}
