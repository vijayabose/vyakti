//! File format readers for extracting text from various document types

use anyhow::{Context, Result};
use std::path::Path;

pub struct FileReader;

impl FileReader {
    /// Extract text from any supported file format
    pub fn read_file(path: &Path) -> Result<String> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        match ext.to_lowercase().as_str() {
            // Text formats
            "md" | "markdown" => Self::read_markdown(path),
            "json" => Self::read_json(path),
            "yaml" | "yml" => Self::read_yaml(path),
            "toml" => Self::read_toml(path),
            "csv" => Self::read_csv(path),
            "html" | "htm" => Self::read_html(path),

            // Documents
            "pdf" => Self::read_pdf(path),
            "ipynb" => Self::read_jupyter(path),
            "docx" => Self::read_docx(path),
            "xlsx" => Self::read_xlsx(path),
            "pptx" => Self::read_pptx(path),

            // Plain text fallback
            "txt" => std::fs::read_to_string(path)
                .with_context(|| format!("Failed to read file: {:?}", path)),

            _ => anyhow::bail!("Unsupported file extension: {}", ext),
        }
    }

    /// Read Markdown file and extract text
    #[cfg(feature = "text-formats")]
    fn read_markdown(path: &Path) -> Result<String> {
        use pulldown_cmark::{Event, Parser, Tag};

        let content = std::fs::read_to_string(path)?;
        let parser = Parser::new(&content);

        let mut text = String::new();
        let mut in_code_block = false;

        for event in parser {
            match event {
                Event::Text(t) | Event::Code(t) => {
                    text.push_str(&t);
                    text.push(' ');
                }
                Event::Start(Tag::CodeBlock(_)) => in_code_block = true,
                Event::End(Tag::CodeBlock(_)) => {
                    in_code_block = false;
                    text.push('\n');
                }
                Event::SoftBreak | Event::HardBreak => {
                    if !in_code_block {
                        text.push(' ');
                    } else {
                        text.push('\n');
                    }
                }
                _ => {}
            }
        }

        Ok(text.trim().to_string())
    }

    #[cfg(not(feature = "text-formats"))]
    fn read_markdown(_path: &Path) -> Result<String> {
        anyhow::bail!("Markdown support requires 'text-formats' feature")
    }

    /// Read JSON file and extract text
    #[cfg(feature = "text-formats")]
    fn read_json(path: &Path) -> Result<String> {
        let content = std::fs::read_to_string(path)?;
        let value: serde_json::Value = serde_json::from_str(&content)?;

        let mut text = String::new();
        Self::extract_json_text(&value, &mut text);

        Ok(text.trim().to_string())
    }

    #[cfg(not(feature = "text-formats"))]
    fn read_json(_path: &Path) -> Result<String> {
        anyhow::bail!("JSON support requires 'text-formats' feature")
    }

    #[cfg(feature = "text-formats")]
    fn extract_json_text(value: &serde_json::Value, output: &mut String) {
        match value {
            serde_json::Value::String(s) => {
                output.push_str(s);
                output.push(' ');
            }
            serde_json::Value::Array(arr) => {
                for item in arr {
                    Self::extract_json_text(item, output);
                }
            }
            serde_json::Value::Object(obj) => {
                for (key, val) in obj {
                    output.push_str(key);
                    output.push_str(": ");
                    Self::extract_json_text(val, output);
                }
            }
            serde_json::Value::Number(n) => {
                output.push_str(&n.to_string());
                output.push(' ');
            }
            serde_json::Value::Bool(b) => {
                output.push_str(&b.to_string());
                output.push(' ');
            }
            _ => {}
        }
    }

    /// Read YAML file and extract text
    #[cfg(feature = "text-formats")]
    fn read_yaml(path: &Path) -> Result<String> {
        let content = std::fs::read_to_string(path)?;
        let value: serde_yaml::Value = serde_yaml::from_str(&content)?;

        let mut text = String::new();
        Self::extract_yaml_text(&value, &mut text);

        Ok(text.trim().to_string())
    }

    #[cfg(not(feature = "text-formats"))]
    fn read_yaml(_path: &Path) -> Result<String> {
        anyhow::bail!("YAML support requires 'text-formats' feature")
    }

    #[cfg(feature = "text-formats")]
    fn extract_yaml_text(value: &serde_yaml::Value, output: &mut String) {
        match value {
            serde_yaml::Value::String(s) => {
                output.push_str(s);
                output.push(' ');
            }
            serde_yaml::Value::Sequence(seq) => {
                for item in seq {
                    Self::extract_yaml_text(item, output);
                }
            }
            serde_yaml::Value::Mapping(map) => {
                for (key, val) in map {
                    if let serde_yaml::Value::String(k) = key {
                        output.push_str(k);
                        output.push_str(": ");
                    }
                    Self::extract_yaml_text(val, output);
                }
            }
            serde_yaml::Value::Number(n) => {
                output.push_str(&n.to_string());
                output.push(' ');
            }
            serde_yaml::Value::Bool(b) => {
                output.push_str(&b.to_string());
                output.push(' ');
            }
            _ => {}
        }
    }

    /// Read TOML file and extract text
    #[cfg(feature = "text-formats")]
    fn read_toml(path: &Path) -> Result<String> {
        let content = std::fs::read_to_string(path)?;
        let value: toml::Value = toml::from_str(&content)?;

        let mut text = String::new();
        Self::extract_toml_text(&value, &mut text);

        Ok(text.trim().to_string())
    }

    #[cfg(not(feature = "text-formats"))]
    fn read_toml(_path: &Path) -> Result<String> {
        anyhow::bail!("TOML support requires 'text-formats' feature")
    }

    #[cfg(feature = "text-formats")]
    fn extract_toml_text(value: &toml::Value, output: &mut String) {
        match value {
            toml::Value::String(s) => {
                output.push_str(s);
                output.push(' ');
            }
            toml::Value::Array(arr) => {
                for item in arr {
                    Self::extract_toml_text(item, output);
                }
            }
            toml::Value::Table(table) => {
                for (key, val) in table {
                    output.push_str(key);
                    output.push_str(": ");
                    Self::extract_toml_text(val, output);
                }
            }
            toml::Value::Integer(n) => {
                output.push_str(&n.to_string());
                output.push(' ');
            }
            toml::Value::Float(f) => {
                output.push_str(&f.to_string());
                output.push(' ');
            }
            toml::Value::Boolean(b) => {
                output.push_str(&b.to_string());
                output.push(' ');
            }
            _ => {}
        }
    }

    /// Read CSV file and extract text
    #[cfg(feature = "text-formats")]
    fn read_csv(path: &Path) -> Result<String> {
        use csv::ReaderBuilder;

        let mut reader = ReaderBuilder::new().has_headers(true).from_path(path)?;

        let mut text = String::new();

        // Add headers
        if let Ok(headers) = reader.headers() {
            for header in headers {
                text.push_str(header);
                text.push(' ');
            }
            text.push('\n');
        }

        // Add rows
        for result in reader.records() {
            let record = result?;
            for field in record.iter() {
                text.push_str(field);
                text.push(' ');
            }
            text.push('\n');
        }

        Ok(text)
    }

    #[cfg(not(feature = "text-formats"))]
    fn read_csv(_path: &Path) -> Result<String> {
        anyhow::bail!("CSV support requires 'text-formats' feature")
    }

    /// Read HTML file and extract text
    #[cfg(feature = "text-formats")]
    fn read_html(path: &Path) -> Result<String> {
        let content = std::fs::read_to_string(path)?;
        let text = html2text::from_read(content.as_bytes(), usize::MAX);
        Ok(text)
    }

    #[cfg(not(feature = "text-formats"))]
    fn read_html(_path: &Path) -> Result<String> {
        anyhow::bail!("HTML support requires 'text-formats' feature")
    }

    /// Read Jupyter notebook and extract text
    #[cfg(feature = "text-formats")]
    fn read_jupyter(path: &Path) -> Result<String> {
        let content = std::fs::read_to_string(path)?;
        let notebook: serde_json::Value = serde_json::from_str(&content)?;

        let mut text = String::new();

        if let Some(cells) = notebook.get("cells").and_then(|c| c.as_array()) {
            for cell in cells {
                let cell_type = cell.get("cell_type").and_then(|t| t.as_str());
                let source = cell.get("source");

                if let Some(source) = source {
                    let cell_text = match source {
                        serde_json::Value::String(s) => s.clone(),
                        serde_json::Value::Array(arr) => arr
                            .iter()
                            .filter_map(|v| v.as_str())
                            .collect::<Vec<_>>()
                            .join(""),
                        _ => continue,
                    };

                    // Add cell type marker
                    if let Some(ct) = cell_type {
                        text.push_str(&format!("[{}]\n", ct));
                    }

                    text.push_str(&cell_text);
                    text.push_str("\n\n");
                }
            }
        }

        Ok(text)
    }

    #[cfg(not(feature = "text-formats"))]
    fn read_jupyter(_path: &Path) -> Result<String> {
        anyhow::bail!("Jupyter notebook support requires 'text-formats' feature")
    }

    /// Read PDF file and extract text
    #[cfg(feature = "document-formats")]
    fn read_pdf(path: &Path) -> Result<String> {
        use pdf_extract::extract_text;
        use std::panic;

        // Wrap PDF extraction in panic handler because pdf-extract can panic on malformed PDFs
        let path_buf = path.to_path_buf();
        let result = panic::catch_unwind(|| extract_text(&path_buf));

        match result {
            Ok(Ok(text)) => Ok(text),
            Ok(Err(e)) => Err(e).with_context(|| format!("Failed to extract text from PDF: {:?}", path)),
            Err(_) => anyhow::bail!(
                "PDF extraction panicked (likely malformed PDF with unsupported Unicode): {:?}",
                path
            ),
        }
    }

    #[cfg(not(feature = "document-formats"))]
    fn read_pdf(_path: &Path) -> Result<String> {
        anyhow::bail!("PDF support requires 'document-formats' feature")
    }

    /// Read Microsoft Word file and extract text
    #[cfg(feature = "document-formats")]
    fn read_docx(path: &Path) -> Result<String> {
        use docx_rs::*;

        let bytes = std::fs::read(path)?;
        let docx = read_docx(&bytes)
            .map_err(|e| anyhow::anyhow!("Failed to read DOCX: {:?}", e))?;

        let mut text = String::new();

        for child in &docx.document.children {
            match child {
                DocumentChild::Paragraph(para) => {
                    for run_child in &para.children {
                        if let ParagraphChild::Run(run) = run_child {
                            for run_child in &run.children {
                                if let RunChild::Text(t) = run_child {
                                    text.push_str(&t.text);
                                }
                            }
                        }
                    }
                    text.push('\n');
                }
                _ => {}
            }
        }

        Ok(text)
    }

    #[cfg(not(feature = "document-formats"))]
    fn read_docx(_path: &Path) -> Result<String> {
        anyhow::bail!("DOCX support requires 'document-formats' feature")
    }

    /// Read Microsoft Excel file and extract text
    #[cfg(feature = "document-formats")]
    fn read_xlsx(path: &Path) -> Result<String> {
        use calamine::{open_workbook, Reader, Xlsx};

        let mut workbook: Xlsx<_> = open_workbook(path)?;
        let mut text = String::new();

        for sheet_name in workbook.sheet_names().to_vec() {
            text.push_str(&format!("[Sheet: {}]\n", sheet_name));

            if let Ok(range) = workbook.worksheet_range(&sheet_name) {
                for row in range.rows() {
                    for cell in row {
                        text.push_str(&cell.to_string());
                        text.push('\t');
                    }
                    text.push('\n');
                }
            }

            text.push('\n');
        }

        Ok(text)
    }

    #[cfg(not(feature = "document-formats"))]
    fn read_xlsx(_path: &Path) -> Result<String> {
        anyhow::bail!("XLSX support requires 'document-formats' feature")
    }

    /// Read Microsoft PowerPoint file and extract text
    #[cfg(feature = "document-formats")]
    fn read_pptx(path: &Path) -> Result<String> {
        use std::fs::File;
        use std::io::Read;
        use zip::ZipArchive;

        let file = File::open(path)?;
        let mut archive = ZipArchive::new(file)?;
        let mut text = String::new();

        // Extract text from all slide XML files
        for i in 0..archive.len() {
            let mut file = archive.by_index(i)?;
            let name = file.name().to_string();

            if name.starts_with("ppt/slides/slide") && name.ends_with(".xml") {
                let mut content = String::new();
                file.read_to_string(&mut content)?;

                // Simple XML text extraction
                let extracted = Self::extract_text_from_xml(&content);
                if !extracted.is_empty() {
                    text.push_str(&format!("[Slide]\n{}\n\n", extracted));
                }
            }
        }

        Ok(text)
    }

    #[cfg(not(feature = "document-formats"))]
    fn read_pptx(_path: &Path) -> Result<String> {
        anyhow::bail!("PPTX support requires 'document-formats' feature")
    }

    #[cfg(feature = "document-formats")]
    fn extract_text_from_xml(xml: &str) -> String {
        // Simple text extraction between <a:t> tags
        let mut text = String::new();

        // Find all <a:t> tags
        let mut start_pos = 0;
        while let Some(start_idx) = xml[start_pos..].find("<a:t") {
            let absolute_start = start_pos + start_idx;

            // Find the end of the opening tag
            if let Some(tag_end) = xml[absolute_start..].find('>') {
                let content_start = absolute_start + tag_end + 1;

                // Find the closing tag
                if let Some(close_idx) = xml[content_start..].find("</a:t>") {
                    let content_end = content_start + close_idx;
                    let content = &xml[content_start..content_end];

                    text.push_str(content);
                    text.push(' ');

                    start_pos = content_end;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::write;
    use tempfile::tempdir;

    #[test]
    #[cfg(feature = "text-formats")]
    fn test_read_markdown() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.md");
        write(&file_path, "# Title\n\nSome **bold** text.").unwrap();

        let text = FileReader::read_file(&file_path).unwrap();
        assert!(text.contains("Title"));
        assert!(text.contains("bold"));
        assert!(text.contains("text"));
    }

    #[test]
    #[cfg(feature = "text-formats")]
    fn test_read_json() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.json");
        write(&file_path, r#"{"key": "value", "number": 42}"#).unwrap();

        let text = FileReader::read_file(&file_path).unwrap();
        assert!(text.contains("key"));
        assert!(text.contains("value"));
        assert!(text.contains("42"));
    }

    #[test]
    #[cfg(feature = "text-formats")]
    fn test_read_yaml() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.yaml");
        write(&file_path, "key: value\nnumber: 42").unwrap();

        let text = FileReader::read_file(&file_path).unwrap();
        assert!(text.contains("key"));
        assert!(text.contains("value"));
        assert!(text.contains("42"));
    }

    #[test]
    #[cfg(feature = "text-formats")]
    fn test_read_toml() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.toml");
        write(&file_path, "key = \"value\"\nnumber = 42").unwrap();

        let text = FileReader::read_file(&file_path).unwrap();
        assert!(text.contains("key"));
        assert!(text.contains("value"));
        assert!(text.contains("42"));
    }

    #[test]
    #[cfg(feature = "text-formats")]
    fn test_read_csv() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.csv");
        write(&file_path, "a,b,c\n1,2,3\n4,5,6").unwrap();

        let text = FileReader::read_file(&file_path).unwrap();
        assert!(text.contains('a'));
        assert!(text.contains('b'));
        assert!(text.contains('c'));
        assert!(text.contains('1'));
        assert!(text.contains('2'));
        assert!(text.contains('3'));
    }

    #[test]
    fn test_unsupported_extension() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.xyz");
        write(&file_path, "content").unwrap();

        let result = FileReader::read_file(&file_path);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Unsupported file extension"));
    }
}
