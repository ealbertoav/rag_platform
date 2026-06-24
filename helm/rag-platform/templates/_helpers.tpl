{{/*
Fully qualified app name — truncated to 63 chars (K8s label limit).
*/}}
{{ define "rag-platform.fullname" -}}
{{- if .Values.fullnameOverride }}
{{ .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{ .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{ printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Chart label: name-version (+ replaced with _ to satisfy label constraints).
*/}}
{{ define "rag-platform.chart" -}}
{{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels applied to every resource.
*/}}
{{ define "rag-platform.labels" -}}
helm.sh/chart: {{ include "rag-platform.chart" . }}
{{ include "rag-platform.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels — used in matchLabels and pod template labels.
Must be stable across upgrades (never change after first deploy).
*/}}
{{ define "rag-platform.selectorLabels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Service account name.
*/}}
{{ define "rag-platform.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{ default (include "rag-platform.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{ default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}
