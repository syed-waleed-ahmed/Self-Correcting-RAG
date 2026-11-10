import type { Metadata } from "next";
import { AgentView } from "@/components/agent-view";

export const metadata: Metadata = {
  title: "Agent",
  description: "A bounded tool-using agent: plan, run tools, write with citations, verify.",
};

export default function AgentPage() {
  return <AgentView />;
}
