import React, { useState, useEffect } from 'react'
import { Card, Table, Tag, Button, Modal, Form, Input, InputNumber, Select, message } from 'antd'
import { PlusOutlined } from '@ant-design/icons'
import { warehouseApi } from '../../api/warehouse'
import { materialApi } from '../../api/material'

export default function IncomingPage() {
  const [materials, setMaterials] = useState<any[]>([])
  const [lots, setLots] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [modalVisible, setModalVisible] = useState(false)
  const [form] = Form.useForm()

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    try {
      const [matRes, lotRes] = await Promise.all([
        materialApi.getMaterials(),
        materialApi.getLots()
      ])
      setMaterials(matRes.data)
      setLots(lotRes.data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = async (values: any) => {
    try {
      message.success('Lot录入成功')
      setModalVisible(false)
      fetchData()
    } catch (err) {
      message.error('录入失败')
    }
  }

  const lotColumns = [
    { title: 'Lot号', dataIndex: 'lot_number', key: 'lot_number' },
    { title: '物料', dataIndex: 'material_id', key: 'material_id' },
    { title: '数量', dataIndex: 'quantity', key: 'quantity' },
    { title: '状态', dataIndex: 'status', key: 'status', render: (s: string) => {
      const colors: Record<string, string> = { active: 'green', used: 'default', quarantined: 'red' }
      const texts: Record<string, string> = { active: '可用', used: '已使用', quarantined: '隔离' }
      return <Tag color={colors[s]}>{texts[s]}</Tag>
    }},
    { title: '录入时间', dataIndex: 'created_at', key: 'created_at' },
  ]

  return (
    <div>
      <Card 
        title="来料Lot录入" 
        extra={<Button type="primary" icon={<PlusOutlined />} onClick={() => setModalVisible(true)}>录入Lot</Button>}
      >
        <Table 
          columns={lotColumns} 
          dataSource={lots}
          loading={loading}
          rowKey="id"
        />
      </Card>

      <Modal title="录入Lot" open={modalVisible} onCancel={() => setModalVisible(false)} onOk={() => form.submit()}>
        <Form form={form} onFinish={handleSubmit} layout="vertical">
          <Form.Item name="lot_number" label="Lot号" rules={[{ required: true }]}>
            <Input placeholder="例如: LOT20240001" />
          </Form.Item>
          <Form.Item name="material_id" label="物料" rules={[{ required: true }]}>
            <Select placeholder="选择物料">
              {materials.map(m => <Select.Option key={m.id} value={m.id}>{m.name}</Select.Option>)}
            </Select>
          </Form.Item>
          <Form.Item name="quantity" label="数量" rules={[{ required: true }]}>
            <InputNumber style={{ width: '100%' }} min={1} />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}
