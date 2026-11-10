import type { Metadata } from "next";
import { DocumentsView } from "@/components/documents-view";

export const metadata: Metadata = {
  title: "My documents",
  description: "Upload your own documents to a private 24-hour session and ask questions about them.",
};

export default function DocumentsPage() {
  return <DocumentsView />;
}
