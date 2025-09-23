import React, { useState } from 'react';

export default function ControlPanel() {
  const [intersection, setIntersection] = useState('int0');
  const [phase, setPhase] = useState(0);

  const apply = async () => {
    await fetch('/api/v1/control/signal', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ intersection_id: intersection, phase })
    });
    alert('Applied');
  };

  return (
    <div className="bg-white p-3 rounded-lg shadow">
      <h2 className="font-semibold">Control</h2>
      <div className="mt-2">Intersection: <input value={intersection} onChange={e => setIntersection(e.target.value)} /></div>
      <div className="mt-2">Phase: <input type="number" value={phase} onChange={e => setPhase(Number(e.target.value))} /></div>
      <button className="mt-2 p-2 border rounded" onClick={apply}>Apply</button>
    </div>
  );
}
