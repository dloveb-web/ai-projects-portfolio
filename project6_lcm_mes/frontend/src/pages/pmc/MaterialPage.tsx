import { useState, useEffect } from 'react'
import { Card, Table, Tag, Progress, Alert } from 'antd'
import { materialApi } from '../../api/material'

export default function MaterialPage() {
  const [materials, setMaterials] = useState<any[]>([])
  const [lots, setLots] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

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

  const materialColumns = [
    { title: '物料编码', dataIndex: 'code', key: 'code' },
    { title: '物料名称', dataIndex: 'name', key: 'name' },
    { title: '类型', dataIndex: 'type', key: 'type', render: (type: string) => {
      const colors: Record<string, string> = {
        glass: 'blue', polarizer: 'green', ic: 'purple', fpc: 'orange', backlight: 'cyan', other: 'default'
      }
      const texts: Record<string, string> = {
        glass: '玻璃基板', polarizer: '偏光片', ic: '驱动IC', fpc: 'FPC', backlight: '背光模组', other: '其他'
      }
      return <Tag color={colors[type]}>{texts[type]}</Tag>
    }},
    { title: '数量', dataIndex: 'quantity', key: 'quantity' },
    { title: '单位', dataIndex: 'unit', key: 'unit' },
    { title: '库存状态', key: 'status', render: (_: any, record: any) => {
      const percent = Math.min(100, (record.quantity / 10000) * 100)
      const status = percent < 20 ? 'exception' : percent < 50 ? 'active' : 'normal'
      return (
        <Progress 
          percent={percent} 
          status={status} 
          size="small"
          format={() => `${record.quantity}${record.unit}`}
        />
      )
    }},
  ]

  const lotColumns = [
    { title: 'Lot号', dataIndex: 'lot_number', key: 'lot_number' },
    { title: '物料', dataIndex: 'material_id', key: 'material_id' },
    { title: '数量', dataIndex: 'quantity', key: 'quantity' },
    { title: '状态', dataIndex: 'status', key: 'status', render: (s: string) => {
      const colors: Record<string, string> = { active: 'green', used: 'default', quarantined: 'red' }
      const texts: Record<string, string> = { active: '可用', used: '已使用', quarantined: '隔离' }
      return <Tag color={colors[s]}>{texts[s]}</Tag>
    }},
  ]

  const lowStockMaterials = materials.filter(m => m.quantity < 2000)

  return (
    <div>
      {lowStockMaterials.length > 0 && (
        <Alert
          message="物料预警"
          description={`以下物料库存不足: ${lowStockMaterials.map(m => m.name).join(', ')}`}
          type="warning"
          showIcon
          style={{ marginBottom: 24 }}
        />
      )}
      
      <Card title="物料库存">
        <Table 
          columns={materialColumns} 
          dataSource={materials}
          loading={loading}
          rowKey="id"
        />
      </Card>

      <Card title="Lot管理" style={{ marginTop: 24 }}>
        <Table 
          columns={lotColumns} 
          dataSource={lots}
          loading={loading}
          rowKey="id"
        />
      </Card>
    </div>
  )
}
