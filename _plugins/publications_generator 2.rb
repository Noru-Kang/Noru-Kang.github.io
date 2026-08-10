module Jekyll
  class PublicationsGenerator < Generator
    safe true
    priority :high

    def generate(site)
      # BibTeX 파일 경로
      bib_file = File.join(site.source, '_data', 'publications.bib')
      
      return unless File.exist?(bib_file)

      begin
        content = File.read(bib_file)
        publications = parse_bibtex(content)
        site.data['publications'] = publications

      rescue StandardError => e
        Jekyll.logger.warn "Error parsing BibTeX file: #{e.message}"
        site.data['publications'] = []
      end
    end

    private

    def parse_bibtex(content)
      publications = []
      
      # BibTeX 항목 파싱
      content.scan(/@(\w+)\s*\{\s*([^,\n]+),\s*([\s\S]*?)\n\s*\}/i) do |entry_type, cite_key, fields_str|
        pub = {
          'type' => entry_type.downcase.strip,
          'cite_key' => cite_key.strip
        }

        # 필드 파싱
        parse_fields(fields_str, pub)
        publications << pub
      end

      publications
    end

    def parse_fields(fields_str, pub)
      # 필드 파싱: key = {value} 또는 key = "value" 형식
      fields_str.scan(/(\w+)\s*=\s*(?:\{([^}]*)\}|"([^"]*)"|(\d+))/i) do |key, brace_value, quote_value, number_value|
        value = brace_value || quote_value || number_value
        pub[key.downcase.strip] = value.strip if value
      end
    end
  end
end
