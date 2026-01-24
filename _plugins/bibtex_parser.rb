module Jekyll
  class BibTexParser
    def self.parse(file_path)
      publications = []
      return publications unless File.exist?(file_path)

      content = File.read(file_path)
      
      # Parse BibTeX entries
      content.scan(/@(\w+)\{([^,]+),\s*(.*?)\n\}/m) do |entry_type, cite_key, fields|
        pub = {
          'type' => entry_type.downcase,
          'cite_key' => cite_key.strip
        }

        # Parse fields
        fields.scan(/(\w+)\s*=\s*\{([^}]*)\}|(\w+)\s*=\s*"([^"]*)"|(\w+)\s*=\s*(\d+)/) do |match|
          key = match[0] || match[2] || match[4]
          value = match[1] || match[3] || match[5]
          
          pub[key.downcase.strip] = value.strip if key
        end

        publications << pub
      end

      publications
    end
  end
end

Liquid::Template.register_filter(Jekyll::BibTexParser)
