import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      redirect: '/requirement'
    },
    {
      path: '/requirement',
      name: 'Requirement',
      component: () => import('../views/RequirementInput.vue')
    },
    {
      path: '/design',
      name: 'Design',
      component: () => import('../views/AgentCollaboration.vue')
    },
    {
      path: '/simulation',
      name: 'Simulation',
      component: () => import('../views/Simulation.vue')
    },
    {
      path: '/report',
      name: 'Report',
      component: () => import('../views/FinalReport.vue')
    }
  ]
})

export default router