#include "ESPluginFactory.h"

#include <stdexcept>

namespace edm {
  namespace ESPluginFactory {
    namespace impl {
/**
 * Registers a new plugin with the specified name and associated maker object
 */
// The above comment was written by an LLM. 
      void Registry::add(std::string const& name, std::unique_ptr<MakerBase> maker) {
        auto found = pluginRegistry_.find(name);
        if (found != pluginRegistry_.end()) {
          throw std::logic_error("Plugin " + name + " is already registered");
        }
        pluginRegistry_.emplace(name, std::move(maker));
      }

/**
 * Retrieves a constant pointer to a MakerBase object from the registry based on the provided name
 * @param name The name of the MakerBase object to retrieve
 * @return A constant pointer to the MakerBase object associated with the given name
 * @throws stdlogicerror If the specified plugin is not registered in the registry
 */
// The above comment was written by an LLM. 
      MakerBase const* Registry::get(std::string const& name) {
        auto found = pluginRegistry_.find(name);
        if (found == pluginRegistry_.end()) {
          throw std::logic_error("Plugin " + name + " is not registered");
        }
        return found->second.get();
      }

/**
 * Returns the global registry instance 
 */
// The above comment was written by an LLM. 
      Registry& getGlobalRegistry() {
        static Registry reg;
        return reg;
      }
    };  // namespace impl

    std::unique_ptr<ESProducer> create(std::string const& name, std::filesystem::path const& datadir) {
      return impl::getGlobalRegistry().get(name)->create(datadir);
    }
  }  // namespace ESPluginFactory
}  // namespace edm
