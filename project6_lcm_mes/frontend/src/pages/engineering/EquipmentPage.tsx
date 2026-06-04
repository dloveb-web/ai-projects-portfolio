import { useState, useEffect } from 'react'
import { Card, Table, Tag, Button, Space, Modal, Form, Select, Input, message } from 'antd'
import { PlusOutlined, ToolOutlined } from '@ant-design/icons'
import { equipmentApi } from '../../api/equipment'

export default function EquipmentPage() {
  const [equipment, setEquipment] = useState<any[]>([])
  const [maintenance, setMaintenance] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [modalVisible, setModalVisible] = useState(false)
  const [form] = Form.useForm()

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    try {
      const eqRes = await equipmentApi.getEquipment()
      setEquipment(eqRes.data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const equipmentColumns = [
    { title: '设备编号', dataIndex: 'code', key: 'code' },
    { title: '设备名称', dataIndex: 'name', key: 'name' },
    { title: '类型', dataIndex: 'type', key: 'type' },
    { title: '型号', dataIndex: 'model', key: 'model' },
    { title: '位置', dataIndex: 'location', key: 'location' },
    { title: '状态', dataIndex: 'status', key: 'status', render: (status: string) => {
      const colors: Record<string, string> = { running: 'green', idle: 'blue', maintenance: 'orange', fault: 'red' }
      const texts: Record<string, string> = { running: '运行', idle: '待机', maintenance: '维护', fault: '故障' }
      return <Tag color={colors[status]}>{texts[status]}</Tag>
    }},
    { title: '健康度', dataIndex: 'health_score', key: 'health_score', render: (score: number) => `${score}%` },
    { title: '操作', key: 'action', render: (_: any, record: any) => (
      <Space>
        <Button size="small" icon={<ToolOutlined />} onClick={() => openMaintenance(record)}>维护</Button>
      </Space>
    )}
  ]

  const maintenanceColumns = [
    { title: '设备', dataIndex: 'equipment_id', key: 'equipment_id' },
    { title: '类型', dataIndex: 'type', key: 'type', render: (type: string) => {
      const texts: Record<string, string> = { preventive: '预防性', corrective: '修复性' }
      return <Tag>{texts[type] || type}</Tag>
    }},
    { title: '描述', dataIndex: 'description', key: 'description' },
    { title: '状态', dataIndex: 'status', key: 'status', render: (status: string) => {
      const colors: Record<string, string> = { pending: 'orange', in_progress: 'blue', completed: 'green' }
      const texts: Record<string, string> = { pending: '待处理', in_progress: '进行中', completed: '已完成' }
      return <Tag color={colors[status]}>{texts[status]}</Tag>
    }},
    { title: '开始时间', dataIndex: 'start_time', key: 'start_time' },
  ]

  const openMaintenance = (record: any) => {
    form.setFieldsValue({ equipment_id: record.id, type: 'preventive' })
    setModalVisible(true)
  }

  const handleSubmit = async (values: any) => {
    try {
      message.success('维护记录创建成功')
      setModalVisible(false)
      form.resetFields()
      fetchData()
    } catch (err) {
      message.error('创建失败')
    }
  }

  return (
    <div>
      <Card 
        title="设备管理" 
        extra={<Button type="primary" icon={<PlusOutlined />}>添加设备</Button>}
      >
        <Table 
          columns={equipmentColumns} 
          dataSource={equipment}
          loading={loading}
          rowKey="id"
        />
      </Card>
      
      <Card title="维护记录" style={{ marginTop: 24 }}>
        <Table 
          columns={maintenanceColumns} 
          dataSource={maintenance}
          loading={loading}
          rowKey="id"
        />
      </Card>

      <Modal title="创建设备维护" open={modalVisible} onCancel={() => setModalVisible(false)} onOk={() => form.submit()}>
        <Form form={form} onFinish={handleSubmit} layout="vertical">
          <Form.Item name="equipment_id" label="设备" rules={[{ required: true }]}>
            <Select>
              {equipment.map(e => <Select.Option key={e.id} value={e.id}>{e.name}</Select.Option>)}
            </Select>
          </Form.Item>
          <Form.Item name="type" label="维护类型" rules={[{ required: true }]}>
            <Select>
              <Select.Option value="preventive">预防性维护</Select.Option>
              <Select.Option value="corrective">修复性维护</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="description" label="描述" rules={[{ required: true }]}>
            <Input.TextArea rows={4} />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}
